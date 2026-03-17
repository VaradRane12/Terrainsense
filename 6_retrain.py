"""
STEP 6 — Retrain
After you've reviewed and corrected batch1 labels, this script:
  1. Splits batch1 into train/val (80/20)
  2. Writes a data.yaml
  3. Fine-tunes your model on the new labels
  4. Reports new precision so you can track improvement

Usage: python 6_retrain.py
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────
LABELED_DIR    = "review/batch1"      # your corrected frames + labels
DATASET_DIR    = "dataset"            # where train/val splits go
BASE_WEIGHTS   = "weights/your_model.pt"
OUTPUT_DIR     = "weights"

CLASSES        = ["ball", "player"]   # ← edit to match your classes.txt
VAL_SPLIT      = 0.2                  # 20% of batch goes to validation
EPOCHS         = 50
IMG_SIZE       = 640
# ─────────────────────────────────────────────────────────


def build_dataset(labeled_dir, dataset_dir, val_split):
    for split in ("train", "val"):
        os.makedirs(f"{dataset_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{dataset_dir}/labels/{split}",  exist_ok=True)

    frames = sorted(Path(labeled_dir).glob("*.jpg"))
    random.shuffle(frames)

    n_val   = max(1, int(len(frames) * val_split))
    val_set = set(f.name for f in frames[:n_val])

    for f in frames:
        split = "val" if f.name in val_set else "train"
        shutil.copy(f, f"{dataset_dir}/images/{split}/{f.name}")
        label = Path(labeled_dir) / (f.stem + ".txt")
        if label.exists():
            shutil.copy(label, f"{dataset_dir}/labels/{split}/{f.stem}.txt")

    n_train = len(frames) - n_val
    print(f"Dataset split → train: {n_train}  |  val: {n_val}")
    return n_train, n_val


def write_yaml(dataset_dir, classes):
    yaml_path = f"{dataset_dir}/data.yaml"
    cfg = {
        "path":  os.path.abspath(dataset_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(classes),
        "names": classes,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"data.yaml written → {yaml_path}")
    return yaml_path


def retrain(base_weights, yaml_path, output_dir, epochs, img_size):
    model = YOLO(base_weights)
    results = model.train(
        data    = yaml_path,
        epochs  = epochs,
        imgsz   = img_size,
        project = output_dir,
        name    = "retrain",
        exist_ok= True,
    )

    best = Path(output_dir) / "retrain" / "weights" / "best.pt"
    print(f"\n{'─'*55}")
    print(f"Training done.")
    print(f"Best weights → {best}")

    # ── Quick precision readout ────────────────────────────
    metrics = results.results_dict
    p  = metrics.get("metrics/precision(B)", None)
    r  = metrics.get("metrics/recall(B)", None)
    m  = metrics.get("metrics/mAP50(B)", None)

    print(f"\nResults:")
    if p:  print(f"  Precision : {p:.3f}  (was 0.53)")
    if r:  print(f"  Recall    : {r:.3f}")
    if m:  print(f"  mAP@50    : {m:.3f}")
    print(f"{'─'*55}")
    print(f"\nNext: update BASE_WEIGHTS to {best}")
    print(f"Then re-run steps 3→4→5 with the new weights for batch 2.")
    return best


if __name__ == "__main__":
    build_dataset(LABELED_DIR, DATASET_DIR, VAL_SPLIT)
    yaml_path = write_yaml(DATASET_DIR, CLASSES)
    retrain(BASE_WEIGHTS, yaml_path, OUTPUT_DIR, EPOCHS, IMG_SIZE)
