# Dice project — experiment and change log

Newest entry first. Every training run and every dataset change gets an entry
(the course grades "≥ 2 training runs with documented changes").

## 2026-11-18 — Project moved to its own repository
- **Model / stage:** project setup
- **Change:** The project left the course repository (`lab05/dice_project/`)
  for this standalone repository. Paths changed: raw data
  `lab05/data/dice_data_sets/` → `data/raw/`, project files at the root, the
  course templates (`train_custom_yolo.py`, `webcam_detection.py`,
  `webcam_capture.py`) copied into `scripts/`. No change to the code logic,
  data, split or run 1 results.
- **Why:** Keep the group's work separate from the professor's repository.
- **Result:** Dataset rebuilt with `prepare_dice_type_dataset.py` from the new
  location: same 36 train / 9 val split (seed 42).
- **Conclusion / next step:** Run 2.

## 2026-11-18 — Run 1: baseline dice-type detector
- **Model / stage:** dice-type detector (model A)
- **Change:** First training run. Uses the course template
  `scripts/train_custom_yolo.py` (extended with `--project`, `--name`,
  `--batch`, `--device`; defaults unchanged).
- **Why:** Get a reference score before changing anything: every later run is
  compared with this one.
- **Settings:** yolo11n.pt (pretrained on COCO), epochs 50 (template default 10
  is too few to converge), imgsz 416, batch 4, CPU, all other settings Ultralytics
  defaults (incl. its default augmentations). Dataset v1: 36 train / 9 val images.
  Command: `.venv\Scripts\python.exe scripts/train_custom_yolo.py --data datasets/dice_type/data.yaml --epochs 50 --project runs/dice_type --name run1_baseline`
- **Result:** `runs/dice_type/run1_baseline/` — 50 epochs in 3 min (CPU).
  Best model on val (9 images, 45 dice): precision 0.837, recall 0.758,
  **mAP50 0.852**, mAP50-95 0.644.

  | Class | Precision | Recall | mAP50 |
  |---|---|---|---|
  | D6  | 0.84 | 0.89 | 0.89 |
  | D8  | 0.76 | **0.36** | **0.56** |
  | D10 | 0.65 | 0.89 | 0.87 |
  | D12 | 0.94 | 0.67 | 0.95 |
  | D20 | 1.00 | 0.99 | 0.995 |

  Re-validated at conf 0.25 (the threshold a live demo uses;
  `runs/dice_type/run1_baseline_val_conf025/`): 34 of 45 dice found with the
  right type (76 %), 2 dice missed, 8 false boxes. **5 of 9 D8 were predicted as D10.**
  Loss curves: train and val losses both still going down, mAP50 still rising
  at epoch 50 → no overfitting yet, the model had not finished learning.
- **Conclusion / next step:** Good start (D20 almost perfect) but D8 vs D10 is
  the main weakness: from above both look like a small diamond, and at
  imgsz 416 a ~60 px die shrinks to ~39 px, losing detail. Candidate changes for
  run 2: imgsz 640 (no shrinking), more epochs (curve still rising), more D8/D10
  images in dataset v2.

## 2026-11-18 — Dataset v1 for the dice-type detector (model A)
- **Model / stage:** data
- **Change:** Converted the raw CVAT export (`data/raw/set1_dice_type`)
  into a standard YOLO dataset in `datasets/dice_type/` with a train/val split
  (`scripts/prepare_dice_type_dataset.py`), and added data-analysis graphs
  (`scripts/analyze_dataset.py` → `analysis/dice_type/`).
- **Why:** The raw export could not be trained on: images (`captures/`) and labels
  (`labels/train/`) were not in paired `images/` + `labels/` folders, and there was
  no validation set. Webcam frames taken seconds apart are near-duplicates, so the
  split puts whole "scenes" (frames < 5 s apart) into either train or val to
  avoid data leakage.
- **Settings:** 45 labelled images (640×480), 225 boxes; 15 scenes; split
  36 train / 9 val (seed 42). 1 image skipped because it has no label
  (`frame_20261117_184714_170925.jpg`).
- **Result (data analysis):**
  - Perfect class balance: every class has 36 train / 9 val boxes — but only
    because every image contains exactly one die of each type.
  - Dice appear 53–91 px wide (median 59 px for D6/D8, ~65–70 px for D10/D12/D20).
  - Dice are mostly in the centre of the image; corners and the left edge are
    almost empty.
  - Labels checked visually (`label_preview.png`): boxes and names are correct.
    One D6 is cut off at the right image edge (29 px box) — correct label.
  - Limitation: all images come from ONE session (same table, same light, same
    purple dice set), so validation scores will be optimistic.
- **Conclusion / next step:** Train run 1 (baseline) on this dataset. For dataset
  v2: capture a second session (other background/light), frames with several
  dice of the same type, fewer than 5 dice, empty frames, and dice near the
  image edges; use that session as validation.
