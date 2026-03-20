import os
import random
import shutil
from pathlib import Path

# -------- paths --------
mixed_folder = "review_rerun2/normal"          # folder with images + labels
dataset_root = "terrain_dataset"       # final dataset

train_ratio = 0.8
valid_ratio = 0.1
test_ratio  = 0.1

# -------- extensions --------
image_ext = {".jpg", ".jpeg", ".png"}

mixed = Path(mixed_folder)

# collect valid image-label pairs
pairs = []

for file in mixed.iterdir():
    if file.suffix.lower() in image_ext:
        label = mixed / (file.stem + ".txt")
        if label.exists():
            pairs.append((file, label))

print("Total valid pairs:", len(pairs))

# shuffle
random.shuffle(pairs)

# split counts
n = len(pairs)
train_n = int(n * train_ratio)
valid_n = int(n * valid_ratio)

train = pairs[:train_n]
valid = pairs[train_n:train_n + valid_n]
test  = pairs[train_n + valid_n:]

# create folders
for split in ["train", "valid", "test"]:
    os.makedirs(f"{dataset_root}/{split}/images", exist_ok=True)
    os.makedirs(f"{dataset_root}/{split}/labels", exist_ok=True)

def move_files(data, split):
    for img, lbl in data:
        shutil.copy(img, f"{dataset_root}/{split}/images/{img.name}")
        shutil.copy(lbl, f"{dataset_root}/{split}/labels/{lbl.name}")

move_files(train, "train")
move_files(valid, "valid")
move_files(test, "test")

print("Train:", len(train))
print("Valid:", len(valid))
print("Test :", len(test))