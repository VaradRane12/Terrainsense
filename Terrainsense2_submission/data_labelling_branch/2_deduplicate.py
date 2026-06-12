"""
STEP 2 — Deduplication
Removes near-identical frames using perceptual hashing (pHash).
Consecutive frames 0.1s apart are often 95%+ identical — this keeps
only visually distinct ones, so you're not reviewing the same scene 10x.

Install: pip install imagehash Pillow
Usage:   python 2_deduplicate.py
"""

import os
import shutil
from pathlib import Path

import imagehash
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────
INPUT_DIR   = "data/raw_frames"
OUTPUT_DIR  = "data/deduped_frames"
HASH_THRESH = 8    # Hamming distance. Lower = stricter (keeps more frames).
                   # 0 = exact duplicates only, 10 = very aggressive dedup.
                   # 8 is a good default for 0.1s intervals.
# ─────────────────────────────────────────────────────────


def deduplicate(input_dir: str, output_dir: str, threshold: int = 8):
    os.makedirs(output_dir, exist_ok=True)

    frames = sorted(Path(input_dir).glob("*.jpg"))
    if not frames:
        print(f"[ERROR] No .jpg files found in {input_dir}")
        return

    print(f"\nDeduplicating {len(frames)} frames (threshold={threshold})...")

    kept   = []
    dropped = 0
    prev_hash = None

    for img_path in frames:
        img  = Image.open(img_path)
        h    = imagehash.phash(img)

        if prev_hash is None or (h - prev_hash) > threshold:
            shutil.copy(img_path, os.path.join(output_dir, img_path.name))
            kept.append(img_path.name)
            prev_hash = h
        else:
            dropped += 1

    print(f"  Kept:    {len(kept)} frames")
    print(f"  Dropped: {dropped} near-duplicates")
    print(f"  Saved to: {output_dir}/")
    print(f"\nReduction: {dropped / len(frames) * 100:.1f}% of frames removed")


if __name__ == "__main__":
    deduplicate(INPUT_DIR, OUTPUT_DIR, HASH_THRESH)
