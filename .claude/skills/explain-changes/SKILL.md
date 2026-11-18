---
name: explain-changes
description: Explain every code change in a student-friendly way. Use EVERY time Claude creates, edits or deletes code or config in this repository — training scripts, YOLO data.yaml files, dataset preparation, data-analysis graphs, notebooks, webcam/demo scripts, or any other file — and also when the user asks "what did you change", "explain this code" or "why". The user is learning Python and machine learning (they know Java) and must be able to understand and defend every line in a project presentation.
---

# Explain every change

The user is a student in the "AI with Python" course (ESIGELEC). They come from Java,
are new to Python and ML, and will be **graded on understanding** ("explain model
choices, trade-offs, metrics" = 25%) and on **documented training iterations**
("≥ 2 training runs with documented changes" = 30%). Code they cannot explain is
worth nothing to them. Every change you make must therefore be understandable
without you in the room.

## 1. While writing the code

- Prefer clear, explicit code over clever code. No dense one-liners, nested
  comprehensions or chained lambdas when a plain `for` loop reads better.
- Use descriptive names (`crop_box_pixels`, not `cb`).
- Put a short comment above each logical block saying **why** it exists, not
  only what it does. Comment every ML-specific value (thresholds, image size,
  epochs, augmentation flags) with the reason it was chosen.
- Every script starts with a docstring: purpose, how to run it (exact command
  from the repository root, using `.venv\Scripts\python.exe` on Windows), its
  inputs and outputs.
- Keep changes small enough to explain. If a change is large, split it into steps
  and explain each one.

## 2. After every change: the explanation

Write the explanation in your reply, using this structure. Skip a section only
if it truly does not apply (for example, no ML concept involved).

```markdown
### What changed
One or two sentences in plain language: the goal of the change.
List each file touched as a clickable link with line numbers, e.g.
[train_custom_yolo.py:12-30](scripts/train_custom_yolo.py#L12-L30).

### How the code works
Walk through the new or changed code block by block, in the order it runs.
Quote the important lines (short snippets, not the whole file) and explain them.
- Explain Python syntax the user may not know (f-strings, list comprehensions,
  `with`, `pathlib.Path`, slicing, unpacking, keyword arguments, decorators).
  When useful, compare with Java ("a dict is like a HashMap", "`with` is like
  try-with-resources").
- Explain each library call you use (Ultralytics, OpenCV, pandas, matplotlib,
  NumPy, PyTorch): what goes in, what comes out, and its type/shape.

### Why it was done this way
The reason behind each important decision, and the alternative you rejected.
Connect it to course concepts where relevant: train/validation split, data
leakage, overfitting, class balance, loss curves, precision/recall, mAP,
confusion matrix, transfer learning / fine-tuning, augmentation, confidence
threshold.

### How to run and check it
The exact command(s) to run, what output to expect, and how to tell whether it
worked (which file appears, which number to look at, what a "good" vs "bad"
result looks like).

### What you should be able to explain
2–4 short questions the user could be asked about this change in the
presentation, each with a one-line answer.
```

### Extra rules by kind of change

**Training a model (new run or changed hyperparameters)**
- State what is different from the previous run and the hypothesis ("more
  epochs because the validation loss was still decreasing at epoch 20").
- Explain every argument passed to `model.train(...)` that is not left at its
  default.
- After the run, explain the outputs Ultralytics writes to `runs/...`:
  `results.png` (train vs val loss curves → over/underfitting),
  `confusion_matrix.png`, `PR_curve.png`, and the metrics precision, recall,
  mAP50, mAP50-95 (detection) or top-1 accuracy (classification). Interpret the
  actual numbers, do not just define them.
- Append an entry to the experiment log (see section 3).

**Graphs and data analysis**
- For each figure, explain: what is on each axis, what one point/bar
  represents, what the user should look for, and what conclusion the current
  data supports (e.g. "D20 has 3× fewer examples than D6 → class imbalance").
- Explain the plotting code line by line the first time a plotting pattern
  appears; later, only explain what is new.

**Dataset / labels / data.yaml changes**
- Explain the folder structure before and after (show a small tree).
- Explain the YOLO label format when relevant:
  `class x_center y_center width height`, all normalised to 0–1.
- Say explicitly how the train/validation split avoids leakage (near-identical
  consecutive webcam frames must not end up in both sets).

**Bug fixes**
- Explain what was wrong, why it produced the observed symptom, and why the fix
  solves it.

## 3. Keep the experiment log

The course grades documented iterations, so every training run and every
dataset change gets an entry in `CHANGES.md` at the repository root
(create it if it does not exist).
Newest entry at the top:

```markdown
## YYYY-MM-DD — <short title>
- **Model / stage:** dice-type detector | face-value classifier | pipeline | data
- **Change:** what was changed compared with the previous version
- **Why:** the hypothesis or problem being addressed
- **Settings:** model, epochs, imgsz, batch, notable augmentations, dataset size
- **Result:** key metrics (and the run folder in `runs/...`), or "not run yet"
- **Conclusion / next step:**
```

Never invent results. If a run has not been executed yet, write "not run yet"
and fill in the result after the user reports it.

## 4. Tone

- Write for a motivated beginner: precise, plain words, define each term the
  first time it appears.
- Be concise: explain what matters for understanding and do not pad. A
  one-line config fix gets a short explanation, a new training pipeline gets
  the full structure.
- If the user asks a follow-up question, answer it, then check whether the
  explanation in the code comments should be improved too.
