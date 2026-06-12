import os
import glob

def polygon_to_bbox(line):
    parts = line.strip().split()
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    
    # extract all x and y coords
    xs = coords[0::2]
    ys = coords[1::2]
    
    # convert to bbox cx, cy, w, h
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min
    
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

# run on all label folders
for folder in ["train", "valid", "test"]:
    label_dir = f"terrain_dataset/{folder}/labels"  # adjust path if needed
    for label_file in glob.glob(f"{label_dir}/*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 5:  # polygon = many points
                new_lines.append(polygon_to_bbox(line))
            else:
                new_lines.append(line.strip())  # already bbox, keep as is
        
        with open(label_file, "w") as f:
            f.write("\n".join(new_lines))

print("Done converting!")