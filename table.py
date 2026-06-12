from collections import Counter
from pathlib import Path

# Dataset root folder
dataset_path = Path("terrain_dataset")

# Label folders
label_dirs = [
    dataset_path / "train" / "labels",
    dataset_path / "valid" / "labels",
    dataset_path / "test" / "labels"
]

# Classes from data.yaml
class_names = {
    0: "Obstacle",
    1: "Person",
    2: "Pothole",
    3: "Vehical"
}

# Count annotations
class_counts = Counter()

for label_dir in label_dirs:

    if not label_dir.exists():
        continue

    for txt_file in label_dir.glob("*.txt"):

        with open(txt_file, "r") as f:

            for line in f:

                if line.strip():

                    class_id = int(line.split()[0])

                    class_counts[class_id] += 1

# Total objects
total = sum(class_counts.values())

# Print distribution table
print("\n3.1 Class Distribution\n")

print(f"{'Class':<15}{'Count':<12}{'Percentage'}")
print("-" * 45)

for class_id in sorted(class_names.keys()):

    count = class_counts[class_id]

    percentage = (count / total * 100) if total > 0 else 0

    print(
        f"{class_names[class_id]:<15}"
        f"{count:<12}"
        f"{percentage:.1f}%"
    )