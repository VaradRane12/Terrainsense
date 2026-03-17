"""
Smart Batch Sampler
Picks ~300 diverse, high-value frames for your first labeling round.

Strategy:
  1. Spread evenly across all 3 clips (temporal diversity)
  2. Bias toward low-confidence frames (model was unsure = more learning signal)
  3. Within each clip, space samples out (avoid near-duplicate scenes)

Usage: python 5_sample_batch.py
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────
PRIORITY_DIR  = "review/priority"   # low-confidence frames
NORMAL_DIR    = "review/normal"     # high-confidence frames
LABELS_DIR    = "data/pseudo_labels"
CLASSES_FILE  = "classes.txt"
BATCH_DIR     = "review/batch1"

TARGET_TOTAL      = 300
PRIORITY_RATIO    = 0.70   # 70% from low-confidence, 30% from normal
# ─────────────────────────────────────────────────────────


def sample_evenly(frames: list, n: int) -> list:
    """Pick n evenly-spaced frames from a list."""
    if len(frames) <= n:
        return frames
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def group_by_clip(frames: list) -> dict:
    by_clip = defaultdict(list)
    for f in frames:
        # stem like "clip1_00042" → key "clip1"
        parts = Path(f).stem.split("_")
        clip_key = parts[0] if parts else "unknown"
        by_clip[clip_key].append(f)
    return dict(by_clip)


def sample_batch(priority_dir, normal_dir, labels_dir, batch_dir, target, priority_ratio):
    os.makedirs(batch_dir, exist_ok=True)

    n_priority = int(target * priority_ratio)
    n_normal   = target - n_priority

    priority_frames = sorted(Path(priority_dir).glob("*.jpg"))
    normal_frames   = sorted(Path(normal_dir).glob("*.jpg"))

    print(f"Available  →  priority: {len(priority_frames)}  |  normal: {len(normal_frames)}")
    print(f"Sampling   →  priority: {n_priority}  |  normal: {n_normal}  |  total: {target}\n")

    sampled = []

    # ── Sample from priority (spread across clips) ─────────────────────
    by_clip = group_by_clip(priority_frames)
    per_clip = n_priority // max(len(by_clip), 1)
    for clip, clip_frames in sorted(by_clip.items()):
        picked = sample_evenly(clip_frames, per_clip)
        sampled.extend(picked)
        print(f"  {clip}: picked {len(picked)} / {len(clip_frames)} priority frames")

    # ── Top up with normal frames if needed ────────────────────────────
    by_clip_n = group_by_clip(normal_frames)
    per_clip_n = n_normal // max(len(by_clip_n), 1)
    for clip, clip_frames in sorted(by_clip_n.items()):
        picked = sample_evenly(clip_frames, per_clip_n)
        sampled.extend(picked)
        print(f"  {clip}: picked {len(picked)} / {len(clip_frames)} normal frames")

    # ── Copy frames + labels into batch dir ────────────────────────────
    copied = 0
    for f in sampled:
        f = Path(f)
        shutil.copy(f, batch_dir)
        label = Path(labels_dir) / (f.stem + ".txt")
        if label.exists():
            shutil.copy(label, batch_dir)
        else:
            open(os.path.join(batch_dir, f.stem + ".txt"), "w").close()
        copied += 1

    if os.path.exists(CLASSES_FILE):
        shutil.copy(CLASSES_FILE, batch_dir)

    print(f"\n{'─'*55}")
    print(f"Batch ready: {copied} frames → {batch_dir}/")
    print(f"{'─'*55}")
    print(f"\nOpen in labelImg:")
    print(f"  labelImg {batch_dir}/ {CLASSES_FILE} {batch_dir}/")
    print(f"\nLabelImg shortcuts:")
    print(f"  W      → save current image")
    print(f"  D / A  → next / previous image")
    print(f"  Del    → delete selected box")
    print(f"  Ctrl+Z → undo")
    print(f"\nAfter labeling, run:  python 6_retrain.py")


if __name__ == "__main__":
    sample_batch(
        PRIORITY_DIR, NORMAL_DIR, LABELS_DIR,
        BATCH_DIR, TARGET_TOTAL, PRIORITY_RATIO
    )
