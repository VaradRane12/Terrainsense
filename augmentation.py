import os
import cv2
import albumentations as A
from tqdm import tqdm

# dataset paths
IMAGE_DIR = "terrain_dataset/train/images"
LABEL_DIR = "terrain_dataset/train/labels"

OUT_IMG = "dataset_aug/train/images"
OUT_LBL = "dataset_aug/train/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# augmentation pipeline
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MotionBlur(p=0.2),
        A.GaussianBlur(p=0.2),
        A.Rotate(limit=10, p=0.5),
        A.RandomShadow(p=0.3),
        A.RandomFog(p=0.2),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)

for img_name in tqdm(os.listdir(IMAGE_DIR)):

    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))

    image = cv2.imread(img_path)

    if image is None:
        continue

    bboxes = []
    labels = []

    if os.path.exists(label_path):

        with open(label_path) as f:
            for line in f.readlines():

                parts = line.strip().split()

                # skip bad labels
                if len(parts) < 5:
                    continue

                try:
                    cls, x, y, w, h = map(float, parts[:5])
                except:
                    continue

                # clamp values to valid YOLO range
                x = max(0, min(1, x))
                y = max(0, min(1, y))
                w = max(0, min(1, w))
                h = max(0, min(1, h))

                bboxes.append([x, y, w, h])
                labels.append(int(cls))

    # generate augmented images
    for i in range(3):

        try:
            transformed = transform(
                image=image,
                bboxes=bboxes,
                class_labels=labels,
            )
        except:
            continue

        aug_img = transformed["image"]
        aug_boxes = transformed["bboxes"]
        aug_labels = transformed["class_labels"]

        new_name = img_name.replace(".jpg", f"_aug{i}.jpg")

        cv2.imwrite(os.path.join(OUT_IMG, new_name), aug_img)

        with open(os.path.join(OUT_LBL, new_name.replace(".jpg", ".txt")), "w") as f:
            for box, label in zip(aug_boxes, aug_labels):
                x, y, w, h = box
                f.write(f"{label} {x} {y} {w} {h}\n")

print("Augmentation finished.")