"""
STEP 1 — Frame Extraction
Extracts 1 frame every 0.1 seconds from each video clip.
Usage: python 1_extract_frames.py
"""

import cv2
import os
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────
CLIPS = [
    "clip1.mp4",   # 4 min  → ~2400 frames
    "clip2.mp4",   # 2 min  → ~1200 frames
    "clip3.mp4",   # 1.5min → ~900  frames
]
OUTPUT_DIR = "data/raw_frames"
INTERVAL_SEC = 0.1   # 1 frame every 100ms = 10 fps effective
# ─────────────────────────────────────────────────────────


def extract_frames(video_path: str, output_dir: str, interval_sec: float = 0.1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(fps * interval_sec)))
    video_name = Path(video_path).stem

    print(f"\n[{video_name}]")
    print(f"  FPS: {fps:.1f} | Total frames: {total_frames} | Saving every {frame_interval} frames")

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            out_path = os.path.join(output_dir, f"{video_name}_{saved_count:05d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"  Saved: {saved_count} frames → {output_dir}/")
    return saved_count


if __name__ == "__main__":
    total = 0
    for clip in CLIPS:
        if not os.path.exists(clip):
            print(f"[SKIP] {clip} not found")
            continue
        total += extract_frames(clip, OUTPUT_DIR, INTERVAL_SEC)

    print(f"\nDone. Total frames extracted: {total}")
    print(f"Frames saved to: {OUTPUT_DIR}/")
