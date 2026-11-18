"""Draw graphs that describe a YOLO detection dataset (data-quality check).

It answers the questions the project is graded on under "data collection &
labeling quality" (volume, consistency, class balance):
  1. class_counts.png    - how many boxes of each dice type, in train and val?
  2. box_sizes.png       - how big are the dice in the picture (pixels)?
  3. box_centers.png     - where in the picture are the dice? (always same spot?)
  4. dice_per_image.png  - how many dice does one picture contain?
  5. label_preview.png   - a few images with their boxes drawn, to check by eye
                           that the labels sit on the right dice.
It also prints a text summary.

How to run (from the repository root, Windows):
    .venv\\Scripts\\python.exe scripts/analyze_dataset.py

Input : a dataset made by prepare_dice_type_dataset.py (default
        datasets/dice_type)
Output: PNG files in analysis/<dataset folder name>/
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw into files only, no window needed
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

# One fixed colour per dice type, always in the same order, so D20 has the same
# colour in every graph. Taken from a colour-blind-checked palette.
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Two colours for the two splits.
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834"}
TEXT_COLOR = "#52514e"
SPLITS = ["train", "val"]


def load_boxes(dataset_folder):
    """Read every label file and return a list of boxes (one dict per box)
    plus the number of boxes in each image.

    A YOLO label line looks like:  2 0.326461 0.484802 0.118703 0.175063
    meaning: class_id, x_center, y_center, width, height. The 4 numbers are
    fractions of the image size (0 = left/top edge, 1 = right/bottom edge)."""
    boxes = []
    dice_per_image = {"train": [], "val": []}

    for split in SPLITS:
        for label_path in sorted((dataset_folder / "labels" / split).glob("*.txt")):
            image_path = dataset_folder / "images" / split / (label_path.stem + ".jpg")
            # Image.open only reads the header here, so getting the size is fast.
            with Image.open(image_path) as image:
                image_width, image_height = image.size

            lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            dice_per_image[split].append(len(lines))

            for line in lines:
                # "2 0.32 0.48 0.11 0.17".split() -> ["2", "0.32", "0.48", "0.11", "0.17"]
                values = line.split()
                boxes.append({
                    "split": split,
                    "image": image_path,
                    "class_id": int(values[0]),
                    "x_center": float(values[1]),
                    "y_center": float(values[2]),
                    # convert width/height from fractions to pixels (easier to imagine)
                    "width_px": float(values[3]) * image_width,
                    "height_px": float(values[4]) * image_height,
                })
    return boxes, dice_per_image


def style_axes(axes):
    """Make the axes quiet so the data stands out: no top/right border,
    light horizontal grid behind the data."""
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(axis="y", color="#e0dfda", linewidth=0.8)
    axes.set_axisbelow(True)  # grid behind the bars/points
    axes.tick_params(colors=TEXT_COLOR)


def plot_class_counts(boxes, class_names, output_folder):
    """Grouped bar chart: one group per dice type, one bar per split."""
    class_ids = sorted(class_names)
    x_positions = np.arange(len(class_ids))  # [0, 1, 2, 3, 4]
    bar_width = 0.38

    figure, axes = plt.subplots(figsize=(7, 4))
    for offset, split in zip([-bar_width / 2, bar_width / 2], SPLITS):
        counts = []
        for class_id in class_ids:
            count = 0
            for box in boxes:
                if box["split"] == split and box["class_id"] == class_id:
                    count += 1
            counts.append(count)
        bars = axes.bar(x_positions + offset, counts, width=bar_width, label=split,
                        color=SPLIT_COLORS[split], edgecolor="white", linewidth=2)
        axes.bar_label(bars, color=TEXT_COLOR, fontsize=9)  # number on top of each bar

    axes.set_xticks(x_positions, [class_names[class_id] for class_id in class_ids])
    axes.set_ylabel("Number of labelled boxes")
    axes.set_title("Boxes per dice type (class balance)", loc="left")
    axes.legend(frameon=False)
    style_axes(axes)
    figure.tight_layout()
    figure.savefig(output_folder / "class_counts.png", dpi=150)
    plt.close(figure)


def plot_box_sizes(boxes, class_names, output_folder):
    """One column of dots per dice type: each dot is one box, its height on the
    y axis is the box size in pixels (square root of width x height)."""
    figure, axes = plt.subplots(figsize=(7, 4))
    # Python's own random module (numpy.random is blocked by Windows Smart App
    # Control on this PC). Seed 0 -> the dots land in the same place every run.
    random_generator = random.Random(0)
    for class_id in sorted(class_names):
        sizes = [np.sqrt(box["width_px"] * box["height_px"])
                 for box in boxes if box["class_id"] == class_id]
        # small random horizontal shift ("jitter") so dots do not hide each other
        x_values = [class_id + random_generator.uniform(-0.15, 0.15) for _ in sizes]
        axes.scatter(x_values, sizes, s=22, color=CLASS_COLORS[class_id],
                     edgecolor="white", linewidth=0.8)
        # black line = median size of this dice type
        axes.hlines(np.median(sizes), class_id - 0.25, class_id + 0.25, color="#0b0b0b", linewidth=2)

    axes.set_xticks(sorted(class_names), [class_names[c] for c in sorted(class_names)])
    axes.set_ylabel("Box size in pixels  (√(width × height))")
    axes.set_title("How big each die appears in the image (line = median)", loc="left")
    axes.set_ylim(bottom=0)
    style_axes(axes)
    figure.tight_layout()
    figure.savefig(output_folder / "box_sizes.png", dpi=150)
    plt.close(figure)


def plot_box_centers(boxes, output_folder):
    """Heat map of where box centres fall in the image (all classes together).
    Dark cell = many dice were in that area of the picture."""
    x_values = [box["x_center"] for box in boxes]
    y_values = [box["y_center"] for box in boxes]

    figure, axes = plt.subplots(figsize=(6, 4.8))
    # histogram2d counts the points in an 8 x 6 grid of cells (same 4:3 shape as the camera)
    counts, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=[8, 6], range=[[0, 1], [0, 1]])
    # .T (transpose) because imshow expects rows = y; origin="upper" puts y=0 at the top like an image
    image = axes.imshow(counts.T, cmap="Blues", extent=[0, 1, 1, 0], aspect=0.75)
    figure.colorbar(image, ax=axes, label="Number of box centres")
    axes.set_xlabel("x (0 = left edge, 1 = right edge)")
    axes.set_ylabel("y (0 = top edge, 1 = bottom edge)")
    axes.set_title("Where the dice are in the picture", loc="left")
    figure.tight_layout()
    figure.savefig(output_folder / "box_centers.png", dpi=150)
    plt.close(figure)


def plot_dice_per_image(dice_per_image, output_folder):
    """Bar chart: how many images contain 0, 1, 2, ... dice."""
    all_counts = dice_per_image["train"] + dice_per_image["val"]
    possible_counts = np.arange(0, max(all_counts) + 2)
    number_of_images = [all_counts.count(n) for n in possible_counts]

    figure, axes = plt.subplots(figsize=(6, 3.6))
    bars = axes.bar(possible_counts, number_of_images, color=SPLIT_COLORS["train"],
                    edgecolor="white", linewidth=2)
    axes.bar_label(bars, color=TEXT_COLOR, fontsize=9)
    axes.set_xticks(possible_counts)
    axes.set_xlabel("Dice in the image")
    axes.set_ylabel("Number of images")
    axes.set_title("Dice per image (train + val)", loc="left")
    style_axes(axes)
    figure.tight_layout()
    figure.savefig(output_folder / "dice_per_image.png", dpi=150)
    plt.close(figure)


def plot_label_preview(boxes, class_names, output_folder, number_of_images=6):
    """Draw the labelled boxes on a few images, spread across the dataset,
    to check visually that every box is on the right die with the right name."""
    all_images = sorted({box["image"] for box in boxes})
    # pick images evenly spread from first to last (not 6 neighbours that look the same)
    chosen_indices = np.linspace(0, len(all_images) - 1, number_of_images).astype(int)
    chosen_images = [all_images[i] for i in chosen_indices]

    # Pillow's built-in font at a readable size (the default one is ~10 px, too small)
    label_font = ImageFont.load_default(size=18)

    figure, axes_grid = plt.subplots(2, 3, figsize=(12, 6.4))
    for axes, image_path in zip(axes_grid.flat, chosen_images):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for box in boxes:
            if box["image"] != image_path:
                continue
            # convert centre + size (pixels) into the corner coordinates PIL needs
            centre_x = box["x_center"] * width
            centre_y = box["y_center"] * height
            left = centre_x - box["width_px"] / 2
            top = centre_y - box["height_px"] / 2
            right = centre_x + box["width_px"] / 2
            bottom = centre_y + box["height_px"] / 2
            color = CLASS_COLORS[box["class_id"]]
            draw.rectangle([left, top, right, bottom], outline=color, width=3)
            # the class name is written as text, so the colour is never the only clue
            draw.text((left, top - 21), class_names[box["class_id"]], fill=color, font=label_font)
        axes.imshow(image)
        axes.set_title(f"{image_path.parent.name}/{image_path.name}", fontsize=8, color=TEXT_COLOR)
        axes.axis("off")
    figure.suptitle("Label check: do the boxes and names match the dice?", x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(output_folder / "label_preview.png", dpi=120)
    plt.close(figure)


def print_summary(boxes, dice_per_image, class_names):
    print("Dataset summary")
    for split in SPLITS:
        print(f"  {split:5s}: {len(dice_per_image[split])} images, "
              f"{sum(dice_per_image[split])} boxes")
    print("  Boxes per class (train / val), median size in pixels:")
    for class_id in sorted(class_names):
        class_boxes = [box for box in boxes if box["class_id"] == class_id]
        train_count = sum(1 for box in class_boxes if box["split"] == "train")
        val_count = sum(1 for box in class_boxes if box["split"] == "val")
        median_size = np.median([np.sqrt(box["width_px"] * box["height_px"]) for box in class_boxes])
        print(f"    {class_names[class_id]:4s} {train_count:3d} / {val_count:3d}   ~{median_size:.0f} px")


def main():
    parser = argparse.ArgumentParser(description="Plot statistics of a YOLO detection dataset.")
    parser.add_argument("--dataset", default="datasets/dice_type")
    parser.add_argument("--output", default="analysis")
    args = parser.parse_args()

    dataset_folder = Path(args.dataset)
    # one sub-folder per dataset, so graphs of dataset v1 and v2 can be compared side by side
    output_folder = Path(args.output) / dataset_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(dataset_folder / "data.yaml", encoding="utf-8") as file:
        class_names = yaml.safe_load(file)["names"]  # dict {0: "D6", 1: "D8", ...}

    boxes, dice_per_image = load_boxes(dataset_folder)
    print_summary(boxes, dice_per_image, class_names)

    plot_class_counts(boxes, class_names, output_folder)
    plot_box_sizes(boxes, class_names, output_folder)
    plot_box_centers(boxes, output_folder)
    plot_dice_per_image(dice_per_image, output_folder)
    plot_label_preview(boxes, class_names, output_folder)
    print(f"Graphs saved in {output_folder}")


if __name__ == "__main__":
    main()
