import os
import yaml

DATASET_DIR = "terrain_dataset"

# new class ids
NEW_CLASSES = {
    "pothole": 0,
    "person": 1,
    "vehicle": 2,
    "obstacle": 3
}

# mapping old class name → new class name
MERGE_MAP = {
    "pothole": "pothole",
    "drain_cover": "pothole",
    "manhole_cover": "pothole",

    "person": "person",

    "car": "vehicle",
    "truck": "vehicle",
    "rickshaw": "vehicle",
    "scooter": "vehicle",
    "bike": "vehicle",
    "parked_scooter": "vehicle",
    "parked_bicycle": "vehicle",

    "bench": "obstacle",
    "pole": "obstacle",
    "tree": "obstacle",
    "tree_trunk": "obstacle",
    "bollard": "obstacle",
    "barrier": "obstacle"
}

# load yaml
with open(os.path.join(DATASET_DIR, "data.yaml")) as f:
    data = yaml.safe_load(f)

old_classes = data["names"]

# build ID mapping
id_map = {}

for old_id, name in enumerate(old_classes):
    if name in MERGE_MAP:
        new_class = MERGE_MAP[name]
        id_map[old_id] = NEW_CLASSES[new_class]

print("Class mapping:", id_map)

splits = ["train", "valid", "test"]

for split in splits:

    label_dir = os.path.join(DATASET_DIR, split, "labels")

    if not os.path.exists(label_dir):
        continue

    for file in os.listdir(label_dir):

        path = os.path.join(label_dir, file)

        new_lines = []

        with open(path) as f:
            for line in f:
                parts = line.strip().split()

                cls = int(parts[0])

                if cls in id_map:
                    parts[0] = str(id_map[cls])
                    new_lines.append(" ".join(parts))

        with open(path, "w") as f:
            for l in new_lines:
                f.write(l + "\n")

print("Done merging classes.")