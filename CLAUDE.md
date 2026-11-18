# Instructions for Claude

- Whenever you create, edit or delete code or configuration in this repository
  (training scripts, data.yaml, dataset tools, analysis graphs, notebooks, demo
  scripts…), follow the `explain-changes` skill in
  `.claude/skills/explain-changes/SKILL.md`: comment the code clearly and
  explain every change in your reply, and log training runs and dataset changes
  in `CHANGES.md`.
- Group project for the ESIGELEC "AI with Python" course: two-stage dice reader
  (YOLO detector for dice type D6/D8/D10/D12/D20, then a model for the number
  showing) with a live webcam demo. This repository contains only the project;
  the course material (slides with the project spec, lab05 templates) is in a
  separate clone of the course repository and must not be pushed to.
- `scripts/train_custom_yolo.py`, `webcam_detection.py` and `webcam_capture.py`
  come from the course's lab05 templates: build on them rather than replacing them.
