import os

label_dir = "terrain_dataset/train/labels"

bad = []

for f in os.listdir(label_dir):
    if not f.endswith(".txt"):
        continue

    path = os.path.join(label_dir, f)

    with open(path) as file:
        for i, line in enumerate(file):
            parts = line.strip().split()

            if len(parts) != 5:
                bad.append((f, "wrong columns"))
                continue

            cls, x, y, w, h = map(float, parts)

            if cls not in [0,1,2,3]:
                bad.append((f, "invalid class"))

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                bad.append((f, "bbox out of range"))

print("Problems:", bad[:20])
print("Total issues:", len(bad))