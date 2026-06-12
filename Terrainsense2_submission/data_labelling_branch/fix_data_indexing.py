import os

paths = [
    "terrain_dataset/train/labels",
    "terrain_dataset/valid/labels",
    "terrain_dataset/test/labels"
]

nc = 4

total_removed = 0
total_files = 0

for labels_path in paths:
    print(f"\nChecking: {labels_path}")

    if not os.path.exists(labels_path):
        print("Path not found:", labels_path)
        continue

    for file in os.listdir(labels_path):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(labels_path, file)

        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        changed = False

        for i, line in enumerate(lines):
            parts = line.strip().split()

            if len(parts) != 5:
                print(f"{file} line {i} → format error:", line.strip())
                total_removed += 1
                changed = True
                continue

            try:
                cls = int(float(parts[0]))
                coords = list(map(float, parts[1:]))

                if cls < 0 or cls >= nc:
                    print(f"{file} line {i} → BAD CLASS:", cls)
                    total_removed += 1
                    changed = True
                    continue

                if not all(0 <= x <= 1 for x in coords):
                    print(f"{file} line {i} → BAD BBOX:", coords)
                    total_removed += 1
                    changed = True
                    continue

                new_lines.append(line)

            except Exception as e:
                print(f"{file} line {i} → parse error:", line.strip())
                total_removed += 1
                changed = True

        if changed:
            total_files += 1
            with open(path, "w") as f:
                f.writelines(new_lines)

print("\nFiles fixed:", total_files)
print("Bad labels removed:", total_removed)