"""
STEP 4 — Review Prep
Organises frames + pseudo-labels for labelImg review.
Priority: flagged (low-confidence) frames first, then the rest.

Creates two review batches:
  review/priority/  ← model was unsure → review these first
  review/normal/    ← model was confident → spot-check these

Also prints the labelImg commands to launch each batch.

Usage: python 4_prep_review.py
"""

import os
import shutil
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────
FRAMES_DIR    = "review_rerun/normal"
LABELS_DIR    = "review_rerun/normal"
FLAGGED_LIST  = "data/priority_review_rerun2.txt"
REVIEW_ROOT   = "review_rerun2"
CLASSES_FILE  = "classes.txt"   # must exist — one class name per line
# ─────────────────────────────────────────────────────────


def copy_pair(frame_src, label_src, dest_dir):
    """Copy frame + its label file to dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(frame_src, dest_dir)

    label_path = Path(label_src) / (Path(frame_src).stem + ".txt")
    if label_path.exists():
        shutil.copy(label_path, dest_dir)
    else:
        # Create an empty label file so labelImg doesn't skip the frame
        open(os.path.join(dest_dir, Path(frame_src).stem + ".txt"), "w").close()

    # Copy classes.txt into batch dir so labelImg picks it up
    if os.path.exists(CLASSES_FILE):
        shutil.copy(CLASSES_FILE, dest_dir)


def prep_review(frames_dir, labels_dir, flagged_list, review_root):
    priority_dir = os.path.join(review_root, "priority")
    normal_dir   = os.path.join(review_root, "normal")

    # Load flagged frame names
    flagged = set()
    if os.path.exists(flagged_list):
        with open(flagged_list) as f:
            flagged = {line.strip() for line in f if line.strip()}

    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        print(f"[ERROR] No frames found in {frames_dir}")
        return

    p_count = n_count = 0
    for frame in frames:
        if frame.name in flagged:
            copy_pair(frame, labels_dir, priority_dir)
            p_count += 1
        else:
            copy_pair(frame, labels_dir, normal_dir)
            n_count += 1

    print(f"\nReview batches ready:")
    print(f"  Priority (low-confidence) : {p_count} frames → {priority_dir}/")
    print(f"  Normal   (high-confidence): {n_count} frames → {normal_dir}/")
    print(f"\n── labelImg commands ──────────────────────────────────────────")
    print(f"  # Review priority frames first:")
    print(f"  labelImg {priority_dir}/ {CLASSES_FILE} {priority_dir}/")
    print(f"")
    print(f"  # Then spot-check normal frames:")
    print(f"  labelImg {normal_dir}/ {CLASSES_FILE} {normal_dir}/")
    print(f"───────────────────────────────────────────────────────────────")
    print(f"\nIn labelImg:")
    print(f"  - Pre-loaded boxes = model's pseudo-labels")
    print(f"  - Fix / delete / add boxes as needed, then Save (W key)")
    print(f"  - Next image: D key | Prev: A key")


if __name__ == "__main__":
    prep_review(FRAMES_DIR, LABELS_DIR, FLAGGED_LIST, REVIEW_ROOT)
