from pathlib import Path
import os

DEDUPED_DIR = "data/deduped_frames"
BATCH_DIR = "review/batch1"
PSEUDO_LABEL_DIR = "data/pseudo_labels"

batch_stems = {p.stem for p in Path(BATCH_DIR).glob("*.jpg")}

deleted_imgs = 0
deleted_labels = 0

for img in Path(DEDUPED_DIR).glob("*.jpg"):
    if img.stem in batch_stems:

        os.remove(img)
        deleted_imgs += 1

        label = Path(PSEUDO_LABEL_DIR) / (img.stem + ".txt")
        if label.exists():
            os.remove(label)
            deleted_labels += 1

print("Deleted images:", deleted_imgs)
print("Deleted pseudo labels:", deleted_labels)