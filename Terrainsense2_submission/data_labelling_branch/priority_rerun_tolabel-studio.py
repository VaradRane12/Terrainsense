"""
Convert YOLO pseudo-labels → Label Studio pre-annotation JSON
so your model's predictions show up as pre-drawn boxes in Label Studio.

Usage: python convert_to_ls.py
Then import the output JSON into Label Studio as predictions.
"""

import os
import json
from pathlib import Path
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────
FRAMES_DIR   = "priority"   # your images
LABELS_DIR   = "priority"   # your .txt pseudo-labels (same folder)
CLASSES_FILE = "classes.txt"
OUTPUT_JSON  = "ls_preannotations.json"
# ─────────────────────────────────────────────────────────


def load_classes(classes_file):
    with open(classes_file) as f:
        return [l.strip() for l in f if l.strip()]


def yolo_to_ls(cx, cy, w, h):
    """Convert YOLO normalized xywh → Label Studio percent x,y,w,h (top-left origin)"""
    x = (cx - w / 2) * 100
    y = (cy - h / 2) * 100
    return round(x, 4), round(y, 4), round(w * 100, 4), round(h * 100, 4)


def convert(frames_dir, labels_dir, classes_file, output_json):
    classes = load_classes(classes_file)
    frames  = sorted(Path(frames_dir).glob("*.jpg"))

    tasks = []
    skipped = 0

    for img_path in frames:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")

        # Get image dimensions for Label Studio
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except Exception:
            skipped += 1
            continue

        # Build predictions list from YOLO txt
        results = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])

                    if cls_id >= len(classes):
                        continue

                    x, y, bw, bh = yolo_to_ls(cx, cy, w, h)

                    results.append({
                        "from_name": "label",
                        "to_name":   "image",
                        "type":      "rectanglelabels",
                        "value": {
                            "x": x, "y": y,
                            "width": bw, "height": bh,
                            "rotation": 0,
                            "rectanglelabels": [classes[cls_id]]
                        }
                    })

        task = {
            "data": {
                "image": f"/data/local-files/?d=priority/{img_path.name}"
            },
            "predictions": [
                {
                    "model_version": "pseudo_v1",
                    "score": 0.5,
                    "result": results
                }
            ] if results else []
        }   
        tasks.append(task)

    with open(output_json, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Converted {len(tasks)} tasks → {output_json}")
    print(f"Skipped:  {skipped} (unreadable images)")
    print(f"\nNext steps:")
    print(f"  1. Open Label Studio → your project → Import")
    print(f"  2. Upload {output_json}")
    print(f"  3. Boxes will appear as pre-annotations on each image")


if __name__ == "__main__":
    convert(FRAMES_DIR, LABELS_DIR, CLASSES_FILE, OUTPUT_JSON)