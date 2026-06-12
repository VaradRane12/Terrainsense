"""
STEP 3 — Pseudo-labeling (Model Inference)
Runs your trained model on all deduped frames and saves predictions
as YOLO .txt label files. Low-confidence frames get flagged for
priority human review.

Supports: Ultralytics YOLO (v5/v8/v11) — swap the load line if you use something else.
Install:  pip install ultralytics
Usage:    python 3_run_inference.py
"""

import os
from pathlib import Path

from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────
MODEL_PATH      = "/Users/varad/Documents/Programming /Terrainsense/runs/detect/train10/weights/best.pt"   # path to your .pt file
FRAMES_DIR      = "review_rerun/normal"
LABELS_DIR      = "review_rerun/normal"      # YOLO .txt files go here
CONF_THRESHOLD  = 0.25   # minimum confidence to save a detection
FLAG_THRESHOLD  = 0.50   # frames where max confidence < this → flagged for priority review
FLAGGED_LIST    = "data/priority_review_rerun2.txt"
# ─────────────────────────────────────────────────────────


def run_inference(model_path, frames_dir, labels_dir, conf_thresh, flag_thresh):
    os.makedirs(labels_dir, exist_ok=True)

    model  = YOLO(model_path)
    frames = sorted(Path(frames_dir).glob("*.jpg"))

    if not frames:
        print(f"[ERROR] No frames found in {frames_dir}")
        return

    print(f"\nRunning inference on {len(frames)} frames...")
    print(f"  Confidence threshold : {conf_thresh}")
    print(f"  Flag threshold       : {flag_thresh}  (frames below this → priority review)\n")

    flagged = []
    total_detections = 0

    for img_path in frames:
        results = model(str(img_path), conf=conf_thresh, verbose=False)[0]

        # ── Save YOLO format labels ────────────────────────────────────
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        boxes = results.boxes

        lines = []
        max_conf = 0.0

        for box in boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            cx, cy, w, h = box.xywhn[0].tolist()   # normalised xywh
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            max_conf = max(max_conf, conf)

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        total_detections += len(lines)

        # ── Flag low-confidence frames ─────────────────────────────────
        # Flag if: no detections at all, OR best detection is shaky
        if max_conf < flag_thresh:
            flagged.append(img_path.name)

    # ── Write priority review list ─────────────────────────────────────
    os.makedirs(os.path.dirname(FLAGGED_LIST), exist_ok=True)
    with open(FLAGGED_LIST, "w") as f:
        f.write("\n".join(flagged))

    print(f"Done.")
    print(f"  Total detections : {total_detections}")
    print(f"  Labels saved to  : {labels_dir}/")
    print(f"  Flagged frames   : {len(flagged)} → {FLAGGED_LIST}")
    print(f"\nReview these first in labelImg — model was least confident on them.")


if __name__ == "__main__":
    run_inference(
        MODEL_PATH,
        FRAMES_DIR,
        LABELS_DIR,
        CONF_THRESHOLD,
        FLAG_THRESHOLD,
    )
