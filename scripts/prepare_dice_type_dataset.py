"""Build a YOLO detection dataset (train + val) for the dice-TYPE detector (model A).

Why this script exists
----------------------
The raw CVAT export in data/raw/set1_dice_type/ cannot be trained on
as it is:
  * the images live in `captures/` while the labels live in `labels/train/`,
    so YOLO cannot pair them (it finds a label by replacing "/images/" with
    "/labels/" in the image path);
  * there is only a training set, no validation set.

This script copies every (image, label) pair into the standard YOLO layout

    <output>/
        data.yaml
        split_report.txt
        images/train/*.jpg   labels/train/*.txt
        images/val/*.jpg     labels/val/*.txt

and chooses the validation images WITHOUT data leakage: webcam frames taken a
few seconds apart look almost identical, so frames are first grouped into
"scenes" (consecutive frames with less than --gap-seconds between them) and
whole scenes are sent to either train or val, never split between both.

The original folder is only READ, never modified.

How to run (from the repository root, Windows):
    .venv\\Scripts\\python.exe scripts/prepare_dice_type_dataset.py

Optional arguments: --source, --output, --val-fraction, --gap-seconds, --seed
(run with --help to see them).
"""
import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def read_capture_time(image_path):
    """Return the capture time stored in the file name, e.g.
    frame_20261117_185157_032488.jpg -> 2026-11-17 18:51:57.032488.

    webcam_capture.py names files with this pattern, so the name tells us when
    the photo was taken (more reliable than the file's 'modified' date, which
    changes when files are copied)."""
    # image_path.stem is the file name without extension: "frame_20261117_185157_032488"
    # split("_", 1) cuts only at the FIRST "_" -> ["frame", "20261117_185157_032488"]
    time_text = image_path.stem.split("_", 1)[1]
    return datetime.strptime(time_text, "%Y%m%d_%H%M%S_%f")


def group_into_scenes(image_paths, gap_seconds):
    """Group images (sorted by time) into scenes.

    A new scene starts whenever more than `gap_seconds` passed since the
    previous photo. Returns a list of scenes; each scene is a list of paths."""
    sorted_paths = sorted(image_paths, key=read_capture_time)

    scenes = []
    current_scene = []
    previous_time = None
    for path in sorted_paths:
        time = read_capture_time(path)
        if previous_time is not None:
            seconds_since_previous = (time - previous_time).total_seconds()
            if seconds_since_previous > gap_seconds:
                scenes.append(current_scene)
                current_scene = []
        current_scene.append(path)
        previous_time = time
    if current_scene:
        scenes.append(current_scene)
    return scenes


def choose_validation_scenes(scenes, val_fraction, seed):
    """Pick whole scenes for validation until about `val_fraction` of the images
    are in validation. The choice is random but reproducible (same seed -> same
    split), so a training run can always be repeated on exactly the same data."""
    total_images = sum(len(scene) for scene in scenes)
    target_val_images = round(total_images * val_fraction)

    # Shuffle a COPY of the scene order with a fixed seed.
    shuffled_indices = list(range(len(scenes)))
    random.Random(seed).shuffle(shuffled_indices)

    val_scene_indices = set()
    val_image_count = 0
    for index in shuffled_indices:
        scene_size = len(scenes[index])
        # Skip scenes that would push validation above the target
        # (e.g. one big burst of 12 frames).
        if val_image_count + scene_size <= target_val_images:
            val_scene_indices.add(index)
            val_image_count += scene_size
        if val_image_count == target_val_images:
            break
    return val_scene_indices


def main():
    parser = argparse.ArgumentParser(description="Create a train/val YOLO dataset for dice types.")
    parser.add_argument("--source", default="data/raw/set1_dice_type",
                        help="Raw CVAT export: contains data.yaml, captures/ and labels/train/")
    parser.add_argument("--output", default="datasets/dice_type",
                        help="Folder to create (its images/ and labels/ are rebuilt each run)")
    # 20 % validation is the usual starting point: enough images to measure the
    # model, while keeping most of our (small) data for learning.
    parser.add_argument("--val-fraction", type=float, default=0.2)
    # Frames taken less than 5 s apart are treated as the same scene
    # (the dice have barely moved, so they are near-duplicates).
    parser.add_argument("--gap-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    image_folder = source / "captures"
    label_folder = source / "labels" / "train"

    # --- 1. Read the class names from the CVAT data.yaml -------------------
    # yaml.safe_load turns the YAML text into Python objects:
    # here a dict like {"names": {0: "D6", 1: "D8", ...}, "path": ".", ...}
    with open(source / "data.yaml", encoding="utf-8") as file:
        source_config = yaml.safe_load(file)
    class_names = source_config["names"]

    # --- 2. Pair every image with its label file -------------------------
    # Two files belong together when they have the same stem (name without extension).
    images_by_stem = {path.stem: path for path in image_folder.glob("*.jpg")}
    labels_by_stem = {path.stem: path for path in label_folder.glob("*.txt")}

    paired_stems = sorted(set(images_by_stem) & set(labels_by_stem))      # in both
    images_without_label = sorted(set(images_by_stem) - set(labels_by_stem))
    labels_without_image = sorted(set(labels_by_stem) - set(images_by_stem))

    print(f"{len(paired_stems)} image/label pairs found")
    for stem in images_without_label:
        # An image with no label file would be treated as "no dice in this
        # picture" by YOLO, which is wrong here, so we leave it out.
        print(f"  SKIPPED (image has no label yet): {stem}.jpg")
    for stem in labels_without_image:
        print(f"  SKIPPED (label has no image):     {stem}.txt")

    # --- 3. Group frames into scenes and choose the validation scenes -----
    paired_images = [images_by_stem[stem] for stem in paired_stems]
    scenes = group_into_scenes(paired_images, args.gap_seconds)
    val_scene_indices = choose_validation_scenes(scenes, args.val_fraction, args.seed)

    # --- 4. Rebuild the output folders ------------------------------------
    # Delete only the folders this script creates, so a re-run never mixes
    # an old split with a new one.
    for sub_folder in ("images", "labels"):
        if (output / sub_folder).exists():
            shutil.rmtree(output / sub_folder)

    report_lines = []
    count_per_split = {"train": 0, "val": 0}
    for scene_number, scene in enumerate(scenes):
        split = "val" if scene_number in val_scene_indices else "train"
        start = read_capture_time(scene[0]).strftime("%H:%M:%S")
        end = read_capture_time(scene[-1]).strftime("%H:%M:%S")
        report_lines.append(f"scene {scene_number:2d}  {start}-{end}  {len(scene):2d} images  -> {split}")

        for image_path in scene:
            label_path = labels_by_stem[image_path.stem]
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)
            # copy2 copies the file content (and its dates); the original stays where it is.
            shutil.copy2(image_path, output / "images" / split / image_path.name)
            shutil.copy2(label_path, output / "labels" / split / label_path.name)
            count_per_split[split] += 1

    # --- 5. Write data.yaml for Ultralytics ---------------------------------
    # No "path:" key on purpose: Ultralytics then uses the folder that contains
    # data.yaml as the dataset root, so the dataset works from any working
    # directory and on every teammate's computer.
    dataset_config = {
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    with open(output / "data.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(dataset_config, file, sort_keys=False)

    # --- 6. Save and print a report (useful for the CHANGES.md log) -------
    summary = (f"train: {count_per_split['train']} images | val: {count_per_split['val']} images "
               f"| {len(scenes)} scenes (gap > {args.gap_seconds}s) | seed {args.seed}")
    report_lines.append(summary)
    (output / "split_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"Dataset written to {output}")


if __name__ == "__main__":
    main()
