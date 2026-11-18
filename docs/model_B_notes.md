# Model B (face value): notes and decisions so far

Model B reads the number showing on each die. It is being trained by another
team member; this page gathers what was found and decided before handing it
over, so nothing has to be rediscovered.

**Status (2026-11-18):** model A (dice type) is finished. **Run 2**
(`runs/dice_type/run2_dataset_v2/weights/best.pt`, imgsz 416) is the model A
used by the pipeline. Run 3 (imgsz 640) was not clearly better and is 2.4× more
work per frame (see `CHANGES.md`).

---

## 1. Pipeline choice: option (b), a classifier on die crops

Both options use two models. The difference is what model B is shown.

**Option (b), chosen:** model B only ever sees one die at a time.

```
  Webcam photo (640×480)
        │
        ▼
  Model A (YOLO detector)      "where are the dice, which type?"  → boxes
        │
        ▼
  Cut each box out of the photo ("crop")  → one small image per die
        │
        ▼  (one crop at a time)
  Model B (YOLO classifier)    "which number is on this die?"     → one answer
        │
        ▼
  Display: "D6: 4   D8: 7   D10: 0   D12: 11   D20: 17"
```

A **classifier** gives one label for the whole image, with no box
(Ultralytics: `yolo11n-cls.pt`); a **detector** (model A) gives boxes.

**Option (a), not chosen:** a second YOLO detector that finds the small
number boxes on the full photo, then the demo matches each number to the die
box it lies in.

| | (a) Number detector on full photo | (b) Classifier on die crops |
|---|---|---|
| What model B sees | whole photo | one die filling the image |
| Fits the current labels directly | yes (boxes are around the numbers) | no: crops must be made first |
| Linking number ↔ die | extra step in the demo | automatic (one crop = one die) |
| Live demo cost | 2 models once per frame | model A once + model B once per die |
| Main risk | numbers are tiny (~21 px) | if model A misses a die, B never sees it |

---

## 2. The raw data: `data/raw/set1_dice_number/`

CVAT export in the **"YOLO 1.1"** format, which is different from the
dice-type sets ("Ultralytics YOLO Detection"):

```text
set1_dice_number/
├── obj.names            class names, one per line (line N = class N)
├── obj.data, train.txt
├── captures/captures/   150 photos (note: nested folder)
└── labels/              97 .txt files, one box per labelled NUMBER
```

### ⚠️ Class number ≠ face value
`obj.names` lists the faces in the order `1, 2, …, 20, 0`, and a label file
stores the **position** in that list:

| Number in the label file | Face on the die |
|---|---|
| `0` | **1** |
| `1` | 2 |
| … | … |
| `19` | 20 |
| `20` | **0** (D10 face) |

Always convert a class number to a face through `obj.names`; never assume
"class = face value".

### What is in it (checked)
- **Boxes are around the number, not the die:** about 22 × 20 px on average,
  against about 77 × 77 px for the dice-type boxes.
- **97 labelled photos** = 44 photos of dice-type set 1 + 53 new photos from
  18/11 that are in no dice-type set. Only those 44 have both a die box and a
  number label.
- **53 photos have no number labels:** the 51 photos of dice-type set 2 + 2
  photos of set 1.
- **375 number boxes**, about 3.9 per photo although each photo shows 5 dice:
  some numbers were not labelled (23 photos have only one box).
- `train.txt` lists 44 images while there are 97 label files: the export looks
  inconsistent, worth checking in CVAT.
- **Very unbalanced classes** (boxes per face): face "1" has 94, faces 15, 16,
  17, 18 have only **5 each**. Expected, since only the D20 (and D12) can show
  high numbers, but 5 examples is very little.

---

## 3. Making the crop dataset (draft code on branch `model-b-draft`)

A working draft, `scripts/prepare_dice_number_dataset.py`, is kept on the
local git branch **`model-b-draft`** (not on `main`). To look at it:
`git switch model-b-draft` (and `git switch main` to come back). What it does,
and what was learned:

1. **Die boxes:** taken from the hand-drawn dice-type labels when the photo
   has them (43 photos), otherwise from model A run 2 at conf 0.25 (54
   photos). The second way is exactly how the live demo crops.
2. **Matching:** a number belongs to the die box that contains the **centre**
   of the number box (closest die centre if two boxes overlap). A die with no
   number is skipped; a die with 2+ numbers would be ambiguous and skipped.
   Result: all 375 number boxes found their die; 24 dice had no number.
3. **Crop:** the die box enlarged by **15 % on every side** (margin 0.15), so
   the whole die stays in the crop. **The live demo must use the same margin.**
4. **Output:** Ultralytics classification layout, one folder per face
   **name** (so the `obj.names` offset cannot leak in):
   `datasets/dice_number/train/17/<photo>_<n>.jpg`, `val/17/…`.
   Ultralytics matches class folders by name and accepts empty folders, so a
   face missing from validation does not shift the others.
5. **Visual check:** a sample of crops showed one whole die per crop with the
   number matching its folder, also for the crops made from model A's boxes.

### Train/validation split: two pitfalls found
- **Leakage:** photos taken seconds apart are near-duplicates, so the split is
  done by **scenes** (photos < 5 s apart stay together), as for model A.
- **Pipeline-safe:** a photo that model A uses for validation should also be
  a model B validation photo (and a model A training photo stays in training),
  so the full pipeline is evaluated on photos neither model has seen.
- **Each face was photographed in one burst** (= one scene). A random scene
  split therefore sends *all* examples of a rare face to one side; the first
  attempt put all 5 crops of face 16 in validation, leaving 0 to learn from.
  Fix used: a scene goes to validation only if every face in it keeps at
  least 5 training crops.

Result of the draft: **375 crops, 283 train / 92 val** (97 photos: 78 / 19),
21 classes. Faces 15 and 17 still had **no validation example** and face 16
only 3 training crops: the real fix is more photos (section 5).

---

## 4. Training notes for model B

- `scripts/train_custom_yolo.py` expects a `data.yaml` **file**: a classifier
  takes the dataset **folder** (`datasets/dice_number`) and the model
  `yolo11n-cls.pt`, so the script needs a small change.
- **Disable horizontal flips** (`fliplr=0.0`). Ultralytics flips training
  images left-right half of the time by default; a mirrored "6", "9" or "17"
  is not a real face.
- Report **top-1 accuracy** and the **confusion matrix** (which faces are
  confused, e.g. 6 ↔ 9, 1 ↔ 7, 16 ↔ 19), and say which faces have no
  validation example, since their accuracy is not measured.
- Use the same `--project` habit as model A: an **absolute** path (the
  training script already makes it absolute). The `yolo` command line puts
  relative paths under Ultralytics' global `runs_dir`, which points to the
  course repository on this laptop.

---

## 5. Getting more photos of the rare faces

Priorities (crops available today):

| Priority | Faces | Crops | Die |
|---|---|---|---|
| 1 | 15, 16, 17, 18 | 5 each | D20 |
| 2 | 10, 14, 19, 20, 0 | 7–9 | D20 (14, 19, 20), D12/D20 (10), D10 (0) |
| 3 | 7, 11, 12, 13 | 10–15 | mixed |

Target: **≥ 15 crops per face, from at least 3 separate sessions**.

1. **Same setup as the demo:** same webcam, height, angle and table. Capture
   with `.venv\Scripts\python.exe scripts\webcam_capture.py --source 0`
   (SPACE = save, Q = quit). Keep the `frame_…` file names: the split reads
   the time from them.
2. **Several rare faces per photo:** set the D20 on a rare face, the D12 on
   10/11/12, the D10 on 0 about half the time (otherwise 7 or 9). Model B
   learns from each die separately.
3. **Three rounds:** one round = the 8 D20 faces (15, 16, 17, 18, 14, 19, 20,
   13), 3 photos each, moving the dice between shots (position in the frame,
   **rotation** of the number, neighbouring die). Round 2: other lighting and
   camera distance. Round 3: other background. 72 photos in total, and each
   face ends up in 3 separate scenes.
4. **Clean up** blurry photos, hidden numbers and accidental duplicates.
5. **Quick extra win:** label the numbers on the 51 dice-type set 2 photos
   (their die boxes already exist): about 250 more crops.

## 6. Labelling in CVAT

- **Use exactly the same labels in the same order as set 1:** `1, 2, …, 20, 0`.
  Best: create the task inside the existing CVAT project (ask the project owner
  to add you as a member), so the labels are inherited.
- In your own CVAT account, either paste the labels in **Raw** mode:
  ```json
  [
  {"name":"1","type":"any","color":"#e6194b","attributes":[]},
  {"name":"2","type":"any","color":"#3cb44b","attributes":[]},
  {"name":"3","type":"any","color":"#ffe119","attributes":[]},
  {"name":"4","type":"any","color":"#4363d8","attributes":[]},
  {"name":"5","type":"any","color":"#f58231","attributes":[]},
  {"name":"6","type":"any","color":"#911eb4","attributes":[]},
  {"name":"7","type":"any","color":"#46f0f0","attributes":[]},
  {"name":"8","type":"any","color":"#f032e6","attributes":[]},
  {"name":"9","type":"any","color":"#bcf60c","attributes":[]},
  {"name":"10","type":"any","color":"#fabebe","attributes":[]},
  {"name":"11","type":"any","color":"#008080","attributes":[]},
  {"name":"12","type":"any","color":"#e6beff","attributes":[]},
  {"name":"13","type":"any","color":"#9a6324","attributes":[]},
  {"name":"14","type":"any","color":"#fffac8","attributes":[]},
  {"name":"15","type":"any","color":"#800000","attributes":[]},
  {"name":"16","type":"any","color":"#aaffc3","attributes":[]},
  {"name":"17","type":"any","color":"#808000","attributes":[]},
  {"name":"18","type":"any","color":"#ffd8b1","attributes":[]},
  {"name":"19","type":"any","color":"#000075","attributes":[]},
  {"name":"20","type":"any","color":"#808080","attributes":[]},
  {"name":"0","type":"any","color":"#000000","attributes":[]}
  ]
  ```
  (this exact content was not confirmed to work; a shorter version without
  `type` and `color` was rejected), **or** add them by hand in
  **Constructor**: *Add label* → type `1` → **Continue** → `2` → **Continue**
  … → `20` → **Continue** → `0` → **Done**. Check the order before *Submit*.
- One box **around the number on the top face**, one box per die. Dice whose
  number is not labelled are simply skipped, so common faces can be left out
  to save time.
- Export as **"YOLO 1.1"** and check that `obj.names` reads `1 … 20, 0`.
- Put the export in `data/raw/set2_dice_number/` with the same layout as
  set 1 (photos in `captures/`, label files in `labels/`).
