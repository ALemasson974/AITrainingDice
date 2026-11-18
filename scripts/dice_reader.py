"""Live dice reader: which dice are on the table, what each one shows, and the total.

Two-stage pipeline (option b, see docs/model_B_notes.md):
    webcam frame -> model A (YOLO detector, models/type_best.pt) finds each die
                    and its type (D6, D8, D10, D12, D20)
                 -> each die is cut out of the frame (a "crop")
                 -> model B (YOLO classifier, models/cls_best.pt) reads the
                    number on that one die
                 -> the window shows a box + "D20: 17" on every die, and a side
                    panel with the list of dice and the total.

Built on the course template scripts/webcam_detection.py: same options
(--source, --conf, --imgsz), same camera loop, same keys (q / ESC to quit) and
the same error handling. The template itself is left unchanged.

How to run (from the repository root, Windows):
    .venv\\Scripts\\python.exe scripts\\dice_reader.py --source 0
    (--source 1 for a second camera, or a video file: --source dice_test.mp4)

Keys in the window: q or ESC = quit, s = save a screenshot in screenshots/.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Faces each type of die can show. Model B reads the number without knowing the
# die type, so its answer can be impossible for the type found by model A
# (e.g. "8" on a D6). Then one of the two models is wrong, and the die is shown
# in orange with a "?" so the user checks it.
# We do NOT force model B to pick an allowed face: tested on the 375 labelled
# dice, forcing made 2 more errors, because when the combination is impossible
# it is usually model A that got the type wrong (a D8 seen as a D6), not B.
# The D10 is numbered 0-9; its "0" counts as 10 (see face_value below).
FACES_BY_TYPE = {
    "D6": {"1", "2", "3", "4", "5", "6"},
    "D8": {"1", "2", "3", "4", "5", "6", "7", "8"},
    "D10": {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
    "D12": {str(number) for number in range(1, 13)},   # "1" ... "12"
    "D20": {str(number) for number in range(1, 21)},   # "1" ... "20"
}

# Colours are (Blue, Green, Red) in OpenCV, not RGB.
COLOR_SURE = (80, 200, 80)       # green: model B is confident
COLOR_UNSURE = (0, 165, 255)     # orange: model B hesitates, check by eye
PANEL_WIDTH = 270                # pixels added on the right for the list + total


def face_value(face_name):
    """Turn the face read by model B into the value that counts in the total.
    The D10 shows 0-9 and its 0 is worth 10. Only the D10 has a "0" face, so a
    "0" always means 10, even if model A got the die type wrong."""
    value = int(face_name)
    if value == 0:
        return 10
    return value


def crop_die(frame, box_xyxy, margin):
    """Cut one die out of the frame, enlarged by `margin` (0.15 = 15 % of the
    box size) on every side. Must match how model B's training crops were
    made (here: tight crops, margin 0 by default).

    frame is a NumPy array of shape (height, width, 3); box_xyxy is
    (x1, y1, x2, y2) in pixels. Slicing frame[top:bottom, left:right] keeps
    the rows top..bottom-1 and columns left..right-1 (like substring in Java)."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    extra_x = (x2 - x1) * margin
    extra_y = (y2 - y1) * margin
    left = max(0, int(x1 - extra_x))
    top = max(0, int(y1 - extra_y))
    right = min(frame_width, int(x2 + extra_x))
    bottom = min(frame_height, int(y2 + extra_y))
    return frame[top:bottom, left:right]


def read_dice(frame, type_model, number_model, args):
    """Run the two models on one frame.

    Returns a list with one dict per die, sorted from left to right:
    {"type": "D20", "face": "17", "value": 17, "type_conf": 0.91,
     "face_conf": 0.88, "consistent": True, "box": (x1, y1, x2, y2)}"""
    # Stage 1: model A finds the dice. [0] = result for our single image.
    detection = type_model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                   device="cpu", verbose=False)[0]
    boxes = detection.boxes.xyxy.tolist()               # [[x1, y1, x2, y2], ...]
    type_indices = detection.boxes.cls.tolist()         # [4.0, 0.0, ...]
    type_confidences = detection.boxes.conf.tolist()    # [0.91, 0.87, ...]
    if not boxes:
        return []

    # Stage 2: cut out every die and let model B read them all in ONE call
    # (passing a list of images is faster than one call per die).
    crops = [crop_die(frame, box, args.margin) for box in boxes]
    readings = number_model.predict(crops, imgsz=args.cls_imgsz, device="cpu", verbose=False)

    dice = []
    for box, type_index, type_conf, reading in zip(boxes, type_indices, type_confidences, readings):
        die_type = type_model.names[int(type_index)]
        # reading.probs.top1 = index of the most likely face class,
        # reading.probs.top1conf = its probability (0-1). names turns the
        # index into the face text, e.g. 9 -> "17".
        face = number_model.names[reading.probs.top1]
        dice.append({
            "type": die_type,
            "face": face,
            "value": face_value(face),
            "type_conf": type_conf,
            "face_conf": float(reading.probs.top1conf),
            "consistent": face in FACES_BY_TYPE[die_type],
            "box": box,
        })
    # Left to right, so the list in the panel matches the table.
    dice.sort(key=lambda die: die["box"][0])
    return dice


def is_unsure(die, min_face_conf):
    """A reading needs checking by eye if model B hesitates, or if the face is
    impossible for the die type (one of the two models is wrong)."""
    return die["face_conf"] < min_face_conf or not die["consistent"]


def draw_results(frame, dice, min_face_conf):
    """Draw a box and "D20: 17" on every die, plus a side panel with the list
    of dice and the total. Returns a new, wider image."""
    annotated = frame.copy()
    for die in dice:
        x1, y1, x2, y2 = (int(value) for value in die["box"])
        color = COLOR_UNSURE if is_unsure(die, min_face_conf) else COLOR_SURE
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        mark = "" if die["consistent"] else " ?"
        label = f"{die['type']}: {die['value']}{mark}"
        # Filled rectangle behind the text so it stays readable on any background.
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - text_height - 8), (x1 + text_width + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Side panel: a dark rectangle glued to the right of the frame.
    # np.hstack puts two images side by side (they must have the same height).
    panel = np.full((annotated.shape[0], PANEL_WIDTH, 3), 30, dtype=np.uint8)
    lines = [(f"Dice on the table: {len(dice)}", (255, 255, 255))]
    for number, die in enumerate(dice, start=1):
        color = COLOR_UNSURE if is_unsure(die, min_face_conf) else COLOR_SURE
        mark = "" if die["consistent"] else " ?"
        lines.append((f"{number}. {die['type']:>3} = {die['value']:>2}{mark}  ({die['face_conf']:.0%})", color))
    total = sum(die["value"] for die in dice)
    lines.append(("", (255, 255, 255)))
    lines.append((f"TOTAL = {total}", (255, 255, 255)))
    unsure_count = sum(1 for die in dice if is_unsure(die, min_face_conf))
    if unsure_count:
        lines.append((f"({unsure_count} to check: orange)", COLOR_UNSURE))

    y = 35
    for text, color in lines:
        scale = 0.9 if text.startswith("TOTAL") else 0.55
        cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if scale > 0.6 else 1)
        y += 40 if scale > 0.6 else 28
    return np.hstack([annotated, panel])


def main() -> int:
    parser = argparse.ArgumentParser(description="Live dice reader: type + value + total")
    parser.add_argument('--source', default='0', help='Camera index (0) or local video file')
    parser.add_argument('--type-model', default='models/type_best.pt', help='Model A: dice-type detector')
    parser.add_argument('--cls-model', default='models/cls_best.pt', help='Model B: face-value classifier')
    # Same default as the template: below 0.25, too many false boxes appear.
    parser.add_argument('--conf', type=float, default=0.25, help='Model A confidence threshold')
    # Must match the image size each model was trained with (A: 416, B: 128).
    parser.add_argument('--imgsz', type=int, default=416, help='Model A image size')
    parser.add_argument('--cls-imgsz', type=int, default=128, help='Model B image size')
    # Margin around each die before it is given to model B (see crop_die).
    # 0 = the box exactly: measured on the 375 labelled dice, margin 0 read
    # 96.3 % correctly, 0.15 -> 93.6 %, 0.25 -> 79.5 % (model B was trained
    # on tight crops).
    parser.add_argument('--margin', type=float, default=0.0)
    # Below this confidence, a reading is shown in orange: check it by eye.
    parser.add_argument('--min-face-conf', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error('--conf must be between 0 and 1')
    for model_path in (args.type_model, args.cls_model):
        if not Path(model_path).is_file():
            parser.error(f'Model not found: {model_path}')
    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, str) and not Path(source).exists():
        parser.error(f'Video not found: {source}')

    type_model = YOLO(args.type_model)
    number_model = YOLO(args.cls_model)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print('Could not open camera/video. Try --source 1 or a sample video path.')
        return 1
    try:
        print('Press q or ESC in the video window to stop, s to save a screenshot.')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dice = read_dice(frame, type_model, number_model, args)
            annotated = draw_results(frame, dice, args.min_face_conf)
            try:
                cv2.imshow('Dice reader (q to quit, s = screenshot)', annotated)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as ex:
                print('OpenCV cannot open a video window in this environment:', ex)
                print('Run from the local desktop, not a headless terminal.')
                return 2
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                Path('screenshots').mkdir(exist_ok=True)
                filename = Path('screenshots') / f'dice_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                cv2.imwrite(str(filename), annotated)
                print('Saved', filename)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
