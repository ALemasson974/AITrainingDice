# AITrainingDice

Group project for the ESIGELEC "AI with Python" course: recognise dice in a live
webcam stream.

1. **Model A — dice type:** a YOLO detector finds each die and says which type it
   is (D6, D8, D10, D12, D20).
2. **Model B — face value:** a second model reads the number showing on each die
   (planned).

The experiment log, with every dataset change and training run, is in
[CHANGES.md](CHANGES.md).

## Folder structure

```text
AITrainingDice/
├── CHANGES.md                 experiment log (training runs, dataset versions)
├── requirements.txt
├── data/raw/                  one folder per photo session (raw CVAT export):
│   ├── set1_dice_type/        webcam photos (captures/) + CVAT labels (labels/train/)
│   └── set2_dice_type/        same layout, second session
├── scripts/
│   ├── prepare_dice_type_dataset.py   raw data -> YOLO dataset with train/val split
│   ├── analyze_dataset.py             data-quality graphs
│   ├── train_custom_yolo.py           training (course template)
│   ├── webcam_detection.py            live demo (course template)
│   └── webcam_capture.py              take new photos (course template)
├── analysis/                  graphs made by analyze_dataset.py
├── docs/                      script explanations (French) + model B notes (model_B_notes.md)
├── datasets/                  generated, not in git
└── runs/                      training results + models, not in git
```

## Setup (Windows, from the repository root)

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Workflow

```bat
:: 1. Build the YOLO dataset (train/val split) from the raw data
::    (default: all sets of the current version; --source picks other sets)
.venv\Scripts\python.exe scripts/prepare_dice_type_dataset.py

:: 2. Look at the data
.venv\Scripts\python.exe scripts/analyze_dataset.py

:: 3. Train (see CHANGES.md for the settings of each run)
.venv\Scripts\python.exe scripts/train_custom_yolo.py --data datasets/dice_type/data.yaml --epochs 50 --project runs/dice_type --name run1_baseline

:: 4. Live demo with the trained model
.venv\Scripts\python.exe scripts/webcam_detection.py --source 0 --model runs/dice_type/run1_baseline/weights/best.pt
```

Trained models (`runs/`) are not stored in git: share `best.pt` by USB or drive.
