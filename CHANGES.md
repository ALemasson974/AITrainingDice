# Dice project — experiment and change log

Newest entry first. Every training run and every dataset change gets an entry
(the course grades "≥ 2 training runs with documented changes").

## 2026-11-18 — Run 3: image size 640 instead of 416
- **Model / stage:** dice-type detector (model A)
- **Change:** Only `imgsz` changed compared with run 2: 640 instead of 416.
  Same dataset v2 (77 train / 19 val) and same validation set, so run 2 and
  run 3 can be compared directly.
- **Why:** In run 2, D8 ↔ D10 is still the main confusion (5 of 38 dice). The
  webcam photos are 640 px wide; at imgsz 416 every image is shrunk to 65 %
  of its size, so a ~60 px die becomes ~39 px and the small differences
  between a D8 and a D10 (number of faces, edge angles) are partly lost.
  Hypothesis: at 640 the network sees the dice at full resolution and confuses
  D8 and D10 less. Expected cost: about 2.4× more pixels per image
  (640² / 416²), so slower training and a slower live demo on CPU.
- **Settings:** yolo11n.pt (pretrained on COCO), epochs 50, **imgsz 640**,
  batch 4, CPU, Ultralytics default augmentations. Dataset v2.
  Command: `.venv\Scripts\python.exe scripts/train_custom_yolo.py --data datasets/dice_type/data.yaml --epochs 50 --imgsz 640 --project runs/dice_type --name run3_imgsz640`
  Live demo with this model: pass `--imgsz 640` too (the demo default is 416).
- **Result:** `runs/dice_type/run3_imgsz640/` — 50 epochs in 13.6 min (CPU),
  2.7× longer than run 2 (5 min); best model = epoch 35. (A first attempt
  crashed at epoch 8 because the dataset folder was deleted during training;
  the dataset was rebuilt identically — same 77/19 split — and run 3 restarted
  from scratch.) Same validation set as run 2 (19 images, 95 dice).

  | On v2 val | Precision | Recall | mAP50 | mAP50-95 |
  |---|---|---|---|---|
  | Run 2 (imgsz 416) | 0.932 | 0.917 | **0.963** | **0.779** |
  | Run 3 (imgsz 640) | **0.938** | **0.968** | 0.957 | 0.766 |

  | Class mAP50 | D6 | D8 | D10 | D12 | D20 |
  |---|---|---|---|---|---|
  | Run 2 | 0.995 | 0.907 | 0.923 | 0.993 | 0.995 |
  | Run 3 | 0.995 | 0.909 | 0.890 | 0.995 | 0.995 |

  At conf 0.25 (`runs/dice_type/run3_imgsz640_val_conf025/`):

  | On v2 val, conf 0.25 | Run 2 | Run 3 |
  |---|---|---|
  | Dice found with the right type | 89 / 95 (94 %) | **91 / 95 (96 %)** |
  | D8 predicted as D10 | 3 of 19 | 2 of 19 |
  | D10 predicted as D8 | 2 of 19 | 1 of 19 |
  | Other wrong types | 1 (D12 → D20) | 1 (D8 → D6) |
  | Dice missed | 0 | 0 |
  | False boxes | 6 | 5 |

  Loss curves: same shape as run 2 — mAP50 ~0.95 from epoch 20, flat after
  that; training box loss keeps falling, validation box loss flat (~0.81–0.85).
- **Conclusion / next step:** No clear gain. At conf 0.25 run 3 gets 2 more
  dice right (D8 ↔ D10 confusions 5 → 3), but mAP50 and mAP50-95 are slightly
  lower; with 19 validation images, one die more or less is ~1 % overall and
  ~5 % per class, so all these differences are within noise. The hypothesis
  "more pixels separate D8 from D10" is **not clearly confirmed**, while the
  cost is real: 2.7× longer training and ~2.4× more work per frame in the live
  demo. Decision: keep **run 2 (imgsz 416) as model A**, because the live demo
  also has to run model B on every die (5 extra predictions per frame), so
  speed matters. To really separate D8 from D10, more D8/D10 photos (and a
  larger validation set) are a better next lever than image size.

## 2026-11-18 — Run 2: same settings, dataset v2 (more data)
- **Model / stage:** dice-type detector (model A)
- **Change:** Only the data changed compared with run 1: dataset v2
  (77 train / 19 val images, set1 + set2) instead of dataset v1 (36 / 9).
  Same model and settings, so any difference comes from the data.
- **Why:** Run 1 learned from only 36 images and confused D8 with D10 (5 of 9
  D8 predicted as D10). Hypothesis: twice as many varied examples of every
  die help the model tell D8 and D10 apart.
- **Settings:** yolo11n.pt (pretrained on COCO), epochs 50, imgsz 416, batch 4,
  CPU, Ultralytics default augmentations (same as run 1). Dataset v2.
  Command: `.venv\Scripts\python.exe scripts/train_custom_yolo.py --data datasets/dice_type/data.yaml --epochs 50 --project runs/dice_type --name run2_dataset_v2`
- **Result:** `runs/dice_type/run2_dataset_v2/` — 50 epochs in 5 min (CPU),
  best model = epoch 35.
  Fair comparison: both models evaluated on the **same** dataset v2 validation
  set (19 images, 95 dice: 19 of each type).

  | | Precision | Recall | mAP50 | mAP50-95 |
  |---|---|---|---|---|
  | Run 1 model on v2 val | 0.754 | 0.849 | 0.837 | 0.648 |
  | **Run 2 model on v2 val** | **0.932** | **0.917** | **0.963** | **0.779** |

  | Class | Run 1 mAP50 | Run 2 mAP50 | Run 2 Precision | Run 2 Recall |
  |---|---|---|---|---|
  | D6  | 0.972 | 0.995 | 0.99 | 1.00 |
  | D8  | 0.643 | **0.907** | 0.88 | 0.74 |
  | D10 | 0.592 | **0.923** | 0.88 | 0.89 |
  | D12 | 0.985 | 0.993 | 0.96 | 0.95 |
  | D20 | 0.995 | 0.995 | 0.95 | 1.00 |

  At conf 0.25 (live-demo threshold; confusion matrices in
  `runs/dice_type/run1_on_v2val_conf025/` and `run2_dataset_v2_val_conf025/`):

  | On v2 val, conf 0.25 | Run 1 | Run 2 |
  |---|---|---|
  | Dice found with the right type | 76 / 95 (80 %) | **89 / 95 (94 %)** |
  | D8 predicted as D10 | 9 of 19 | **3 of 19** |
  | D10 predicted as D8 | 3 of 19 | 2 of 19 |
  | Other wrong types | 5 | 1 (a D12 seen as D20) |
  | Dice missed | 2 | **0** |
  | False boxes (box where there is no die) | 17 | **6** |

  Comparison caveat: 4 of the 19 v2 validation images were in run 1's
  *training* set (dataset v1 split differently), so run 1's score here is
  slightly too optimistic. Run 2 still wins clearly, so its real advantage is,
  if anything, a bit larger than shown.
  Loss curves: mAP50 reaches ~0.95 at epoch 25 and then stays flat
  (0.95–0.96); mAP50-95 peaks at epoch 35 (0.779) and drifts down to 0.739 at
  epoch 50 while the training box loss keeps falling and the validation box
  loss stays flat (~0.81–0.86) → the model has finished learning from this
  data, with first signs of overfitting at the end.
- **Conclusion / next step:** The hypothesis is confirmed: more data improved
  every metric, and D8/D10 mAP50 went from 0.64/0.59 to 0.91/0.92. D8 ↔ D10 is
  still the main confusion (5 of the 38 D8 + D10 dice), and half of the
  false boxes (3 of 6) are labelled D10.
  More epochs alone would not help (plateau since epoch 25). Run 3 candidate:
  imgsz 640 on the same dataset v2 (one change only), to give the network more
  pixels to distinguish D8 from D10. Note: with 19 validation images, one die
  more or less moves recall by ~5 % per class, so small differences are noise.

## 2026-11-18 — Dataset v2 for the dice-type detector (model A): set 2 added
- **Model / stage:** data
- **Change:** Added a second photo session, `data/raw/set2_dice_type/`
  (51 new labelled webcam photos taken on 2026-11-18, CVAT export with the same
  class order D6, D8, D10, D12, D20). Every photo shows all 5 dice, so each
  class gets +51 boxes (dataset v1: 45 boxes per class, v2: 96 per class).
  `scripts/prepare_dice_type_dataset.py` now merges several raw folders
  (`--source` takes one or more; default = set1 + set2). It stops with an
  error if two exports have a different class order or if the same photo is
  labelled in two sets.
  Cleaning before the merge: the set 2 folder also contained 45 byte-identical
  copies of set 1 photos (without labels) and the images were nested in
  `captures/captures/`; the copies were removed and the images moved to
  `captures/` so every set has the same layout.
- **Why:** Run 1 had only 45 images (36 train / 9 val) and confused D8 with
  D10; more varied examples of every die should help the model generalise.
  Each session stays in its own folder so we can always say what each dataset
  version contains.
- **Settings:** 96 images: train 77 / val 19 (6 from set 1, 13 from set 2),
  38 scenes (gap > 5 s), seed 42. Validation still holds whole scenes only
  (no near-duplicate frames shared between train and val).
  Dataset v1 can still be rebuilt exactly (checked: same 36/9 split, same
  files) with `--source data/raw/set1_dice_type`.
- **Result:** trained in run 2 (see the entry above).
- **Note for comparing runs:** adding scenes changes the random split, so the
  v2 validation set is not the same as run 1's. To compare fairly, also
  validate the run 1 model on the v2 validation set.
- **Conclusion / next step:** run 2 = same settings as run 1 (yolo11n, 50
  epochs, imgsz 416) on dataset v2, so the only change is the data.

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
