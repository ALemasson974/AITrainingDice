"""Build the CLASSIFICATION dataset of die crops for the face-value model (model B).

Option (b) of the two-stage pipeline
------------------------------------
    webcam photo -> model A (YOLO detector) finds each die
                 -> each die is cut out of the photo (a "crop")
                 -> model B (YOLO classifier) reads the number on that one die.

Model B only ever sees ONE die per image, so its training data must look the
same: one crop per die, stored in a folder named after the face value.

Inputs
------
* data/raw/set1_dice_number/ (CVAT "YOLO 1.1" export):
      obj.names        face names, one per line: line N is class N
                       (here "1", "2", ..., "20", "0", so class 0 = face "1"!)
      captures/*.jpg   photos (640x480)
      labels/*.txt     one box per labelled NUMBER (drawn around the digits)
* the die boxes for the same photos, taken from:
      1. the dice-type labels (data/raw/set*_dice_type/labels/train/) when the
         photo was labelled there: hand-drawn, the most reliable;
      2. otherwise model A's predictions (--die-model): that is also how the
         crops will be made in the live demo.
* datasets/dice_type/images/ (made by prepare_dice_type_dataset.py): a photo
  that model A uses for validation must also be a validation photo here.

Output (Ultralytics classification layout: one sub-folder per class)
------
    datasets/dice_number/
        train/<face>/<photo>_<die number>.jpg     e.g. train/17/frame_..._2.jpg
        val/<face>/...
        split_report.txt

How it is run (from the repository root, Windows)
-------------------------------------------------
    .venv\\Scripts\\python.exe scripts/prepare_dice_number_dataset.py

Run prepare_dice_type_dataset.py first (this script reads its split).
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

# Reuse the scene grouping of the dice-type script: same protection against
# near-duplicate frames ending up in both train and val (data leakage).
# This import works because Python adds the folder of the script being run
# (scripts/) to the places it looks for modules.
from prepare_dice_type_dataset import group_into_scenes


def read_yolo_boxes(label_path):
    """Read a YOLO label file.

    Returns a list of (class_id, x_center, y_center, width, height); the four
    numbers are normalised to 0-1 (divided by the image width/height)."""
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue  # skip empty lines
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def box_contains_point(die_box, x, y):
    """True if the point (x, y) lies inside the die box (all values 0-1)."""
    x_center, y_center, width, height = die_box
    return abs(x - x_center) <= width / 2 and abs(y - y_center) <= height / 2


def match_numbers_to_dice(die_boxes, number_boxes):
    """Find which number box belongs to which die.

    A number belongs to the die box that contains the CENTRE of the number box.
    When two die boxes overlap and both contain it, the die whose centre is
    closest wins.

    Returns (numbers_per_die, unmatched_count):
      numbers_per_die[i] = list of face classes found inside die i
      unmatched_count    = number boxes that are inside no die box"""
    numbers_per_die = []
    for _ in die_boxes:  # "_" = a loop variable we do not need
        numbers_per_die.append([])
    unmatched_count = 0

    for class_id, x, y, _width, _height in number_boxes:
        best_die = None
        best_distance = None
        for die_index, die_box in enumerate(die_boxes):
            if not box_contains_point(die_box, x, y):
                continue
            # Squared distance between the two centres (no square root needed
            # just to compare which one is smaller).
            distance = (die_box[0] - x) ** 2 + (die_box[1] - y) ** 2
            if best_distance is None or distance < best_distance:
                best_die = die_index
                best_distance = distance
        if best_die is None:
            unmatched_count += 1
        else:
            numbers_per_die[best_die].append(class_id)
    return numbers_per_die, unmatched_count


def crop_die(image, die_box, margin):
    """Cut one die out of the photo, with an extra `margin` on every side.

    The margin (0.15 = the box grows by 15 % of its size on each side) keeps
    the whole die in the crop even when the box is drawn or predicted a bit
    too tight. The live demo must use the SAME margin, so that model B sees
    crops that look like the ones it was trained on."""
    image_width, image_height = image.size  # PIL gives (width, height)
    x_center, y_center, width, height = die_box
    half_width = width * (0.5 + margin)
    half_height = height * (0.5 + margin)
    # Convert from 0-1 to pixels and keep the crop inside the photo.
    left = max(0, round((x_center - half_width) * image_width))
    top = max(0, round((y_center - half_height) * image_height))
    right = min(image_width, round((x_center + half_width) * image_width))
    bottom = min(image_height, round((y_center + half_height) * image_height))
    return image.crop((left, top, right, bottom))


def count_faces(crops):
    """Count how many crops of each face are in a list of (face, die_box) crops.
    Returns a dict like {"17": 2, "4": 1}."""
    counts = {}
    for face, _die_box in crops:
        counts[face] = counts.get(face, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Create the die-crop dataset for the face-value classifier.")
    parser.add_argument("--number-source", default="data/raw/set1_dice_number",
                        help="CVAT YOLO 1.1 export with obj.names, captures/ and labels/")
    parser.add_argument("--type-sources", nargs="+",
                        default=["data/raw/set1_dice_type", "data/raw/set2_dice_type"],
                        help="Dice-type exports whose hand-drawn die boxes are used when available")
    parser.add_argument("--type-dataset", default="datasets/dice_type",
                        help="Dice-type dataset built by prepare_dice_type_dataset.py (to reuse its split)")
    parser.add_argument("--die-model", default="runs/dice_type/run2_dataset_v2/weights/best.pt",
                        help="Model A, used to find the dice on photos without dice-type labels")
    # Must match the image size model A was trained with (run 2: 416).
    parser.add_argument("--die-imgsz", type=int, default=416)
    # Same confidence threshold as the live demo. A false die box does no harm
    # here: it contains no number box, so it is skipped.
    parser.add_argument("--die-conf", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--output", default="datasets/dice_number")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--gap-seconds", type=float, default=5.0)
    # Each face was photographed in ONE burst (= one scene). Moving that scene
    # to validation would leave the model nothing to learn the face from, so a
    # scene only goes to validation if every face in it keeps at least this
    # many training crops.
    parser.add_argument("--min-train-per-face", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    number_source = Path(args.number_source)
    output = Path(args.output)

    # --- 1. Face names --------------------------------------------------------
    # obj.names has one name per line; the line number is the class number.
    # The folder names of the output are the FACE names ("17"), not the class
    # numbers, so the off-by-one order of obj.names cannot cause mistakes later.
    face_names = (number_source / "obj.names").read_text(encoding="utf-8").split()
    print(f"{len(face_names)} faces: {face_names}")

    # --- 2. Photos that have number labels ------------------------------------
    images_by_stem = {path.stem: path for path in (number_source / "captures").glob("*.jpg")}
    labels_by_stem = {path.stem: path for path in (number_source / "labels").glob("*.txt")}
    photo_stems = sorted(set(images_by_stem) & set(labels_by_stem))
    unlabelled_count = len(set(images_by_stem) - set(labels_by_stem))
    print(f"{len(photo_stems)} photos with number labels "
          f"({unlabelled_count} photos without number labels are skipped)")

    # --- 3. Split of the dice-type dataset (which photos model A validates on) -
    type_dataset = Path(args.type_dataset)
    if not (type_dataset / "images" / "val").is_dir():
        raise SystemExit(f"ERROR: {type_dataset}/images/val not found. "
                         f"Run scripts/prepare_dice_type_dataset.py first.")
    type_split_by_stem = {}
    for split in ("train", "val"):
        for path in (type_dataset / "images" / split).glob("*.jpg"):
            type_split_by_stem[path.stem] = split

    # --- 4. Find the die boxes and match each number to its die ---------------
    # Hand-drawn die boxes when the photo has dice-type labels, else model A.
    type_label_by_stem = {}
    for type_source in args.type_sources:
        for path in (Path(type_source) / "labels" / "train").glob("*.txt"):
            type_label_by_stem[path.stem] = path
    die_model = None
    if any(stem not in type_label_by_stem for stem in photo_stems):
        from ultralytics import YOLO  # imported here: slow, and not always needed
        die_model = YOLO(args.die_model)

    crops_by_stem = {}  # photo -> list of (face, die_box): one entry per crop to make
    photos_with_hand_boxes = 0
    photos_with_model_boxes = 0
    dice_without_number = 0
    ambiguous_dice = 0
    unmatched_numbers = []  # (photo, count): numbers that fell in no die box

    for stem in photo_stems:
        if stem in type_label_by_stem:
            die_boxes = []
            for _class_id, x, y, width, height in read_yolo_boxes(type_label_by_stem[stem]):
                die_boxes.append((x, y, width, height))
            photos_with_hand_boxes += 1
        else:
            result = die_model.predict(str(images_by_stem[stem]), imgsz=args.die_imgsz,
                                       conf=args.die_conf, device="cpu", verbose=False)[0]
            # boxes.xywhn = boxes as (x_center, y_center, width, height), normalised 0-1,
            # the same format as a YOLO label file. .tolist() turns the tensor into Python lists.
            die_boxes = [tuple(box) for box in result.boxes.xywhn.tolist()]
            photos_with_model_boxes += 1

        numbers_per_die, unmatched = match_numbers_to_dice(die_boxes, read_yolo_boxes(labels_by_stem[stem]))
        if unmatched:
            unmatched_numbers.append((stem, unmatched))

        crops = []
        for die_index, numbers in enumerate(numbers_per_die):
            if len(numbers) == 0:
                dice_without_number += 1  # number not labelled / not readable
            elif len(numbers) > 1:
                ambiguous_dice += 1  # two numbers on one die: we cannot know which is on top
            else:
                crops.append((face_names[numbers[0]], die_boxes[die_index]))
        crops_by_stem[stem] = crops

    # --- 5. Train/val split, by scenes ----------------------------------------
    # Rule 1: a photo that model A already uses keeps model A's split. Then
    #         evaluating the whole pipeline on validation photos never uses a
    #         photo that either model was trained on.
    # Rule 2: the other scenes are shuffled with a fixed seed and go to
    #         validation until about val_fraction of the photos are in it ...
    # Rule 3: ... but only if every face in the scene keeps at least
    #         --min-train-per-face training crops.
    photo_paths = [images_by_stem[stem] for stem in photo_stems]
    scenes = group_into_scenes(photo_paths, args.gap_seconds)
    target_val_photos = round(len(photo_paths) * args.val_fraction)

    # Faces in each scene, e.g. {"16": 3, "4": 1}
    faces_per_scene = []
    for scene in scenes:
        scene_crops = []
        for path in scene:
            scene_crops += crops_by_stem[path.stem]
        faces_per_scene.append(count_faces(scene_crops))

    # Start with everything in train, then move scenes to val.
    train_crops_per_face = count_faces([crop for crops in crops_by_stem.values() for crop in crops])
    split_by_scene = {}
    free_scene_indices = []
    val_photo_count = 0
    for scene_index, scene in enumerate(scenes):
        type_splits = {type_split_by_stem.get(path.stem) for path in scene}  # a set
        if "val" in type_splits:  # rule 1
            split_by_scene[scene_index] = "val"
            val_photo_count += len(scene)
            for face, count in faces_per_scene[scene_index].items():
                train_crops_per_face[face] -= count
        elif "train" in type_splits:  # rule 1
            split_by_scene[scene_index] = "train"
        else:
            free_scene_indices.append(scene_index)

    random.Random(args.seed).shuffle(free_scene_indices)
    for scene_index in free_scene_indices:
        fits_target = val_photo_count + len(scenes[scene_index]) <= target_val_photos  # rule 2
        keeps_enough_train = True  # rule 3
        for face, count in faces_per_scene[scene_index].items():
            if train_crops_per_face[face] - count < args.min_train_per_face:
                keeps_enough_train = False
        if fits_target and keeps_enough_train:
            split_by_scene[scene_index] = "val"
            val_photo_count += len(scenes[scene_index])
            for face, count in faces_per_scene[scene_index].items():
                train_crops_per_face[face] -= count
        else:
            split_by_scene[scene_index] = "train"

    # --- 6. Rebuild the output folders, one sub-folder per face, per split ----
    # Every face gets a folder in BOTH splits, even when it stays empty, so it
    # is obvious in the explorer which faces have no validation example.
    for split in ("train", "val"):
        if (output / split).exists():
            shutil.rmtree(output / split)
        for face in face_names:
            (output / split / face).mkdir(parents=True, exist_ok=True)

    # --- 7. Cut out and save the crops ----------------------------------------
    crop_count = {}  # (split, face) -> number of crops
    for scene_index, scene in enumerate(scenes):
        split = split_by_scene[scene_index]
        for image_path in scene:
            image = Image.open(image_path).convert("RGB")
            for crop_number, (face, die_box) in enumerate(crops_by_stem[image_path.stem]):
                crop = crop_die(image, die_box, args.margin)
                crop.save(output / split / face / f"{image_path.stem}_{crop_number}.jpg", quality=95)
                key = (split, face)
                crop_count[key] = crop_count.get(key, 0) + 1

    # --- 8. Report --------------------------------------------------------------
    report_lines = [f"{'face':>4} {'train':>6} {'val':>5}"]
    # Sort the faces as numbers (0, 1, 2, ... 20), not as text ("1", "10", "11", ...).
    for face in sorted(face_names, key=int):
        train_crops = crop_count.get(("train", face), 0)
        val_crops = crop_count.get(("val", face), 0)
        warning = ""
        if train_crops < args.min_train_per_face:
            warning = "   <- TOO FEW TRAINING EXAMPLES"
        elif val_crops == 0:
            warning = "   <- no validation example"
        report_lines.append(f"{face:>4} {train_crops:>6} {val_crops:>5}{warning}")

    total_train = sum(count for (split, _face), count in crop_count.items() if split == "train")
    total_val = sum(count for (split, _face), count in crop_count.items() if split == "val")
    train_photos = len(photo_paths) - val_photo_count
    report_lines += [
        "",
        f"photos: train {train_photos} | val {val_photo_count} | {len(scenes)} scenes "
        f"(gap > {args.gap_seconds}s) | seed {args.seed} | min {args.min_train_per_face} train crops per face",
        f"die boxes: hand-drawn on {photos_with_hand_boxes} photos, "
        f"model A ({args.die_model}, conf {args.die_conf}) on {photos_with_model_boxes} photos",
        f"crops: train {total_train} | val {total_val} | margin {args.margin}",
        f"skipped dice: {dice_without_number} without a number label, {ambiguous_dice} with 2+ numbers",
        f"number boxes inside no die box: {sum(count for _stem, count in unmatched_numbers)} "
        f"(on {len(unmatched_numbers)} photos)",
    ]
    for stem, count in unmatched_numbers:
        report_lines.append(f"    {stem}: {count}")

    (output / "split_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"Dataset written to {output}")


if __name__ == "__main__":
    main()
