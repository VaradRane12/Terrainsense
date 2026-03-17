# Model-Assisted Labeling Pipeline

## Install dependencies
```bash
pip install opencv-python imagehash Pillow ultralytics
pip install labelImg   # the annotation tool
```

---

## Folder structure (after running all steps)

```
project/
├── clip1.mp4
├── clip2.mp4
├── clip3.mp4
├── classes.txt                  ← one class name per line
├── weights/
│   └── your_model.pt
│
├── data/
│   ├── raw_frames/              ← Step 1 output  (~4500 frames)
│   ├── deduped_frames/          ← Step 2 output  (fewer, diverse frames)
│   ├── pseudo_labels/           ← Step 3 output  (YOLO .txt per frame)
│   └── priority_review.txt      ← Step 3 output  (low-confidence frame list)
│
└── review/
    ├── priority/                ← Step 4: review THESE FIRST
    └── normal/                  ← Step 4: spot-check these
```

---

## Run order

```bash
python 1_extract_frames.py    # ~2 min  — pulls frames from all 3 clips
python 2_deduplicate.py       # ~3 min  — drops near-identical frames
python 3_run_inference.py     # ~5 min  — generates pseudo-labels + flags low-conf
python 4_prep_review.py       # ~1 min  — organises batches for labelImg
```

Then open labelImg and start with the priority batch:
```bash
labelImg review/priority/
```

---

## Tuning notes

| Parameter | File | Default | Change if... |
|-----------|------|---------|-------------|
| `INTERVAL_SEC` | step 1 | 0.1s | Action is fast → lower; clip is slow → higher |
| `HASH_THRESH`  | step 2 | 8    | Too many dupes kept → raise; too many dropped → lower |
| `CONF_THRESHOLD` | step 3 | 0.25 | Missing detections → lower; too much noise → raise |
| `FLAG_THRESHOLD` | step 3 | 0.50 | Controls how many frames get priority review |

---

## After labeling → retrain loop

1. Copy reviewed frames + corrected labels into `data/images/train/` and `data/labels/train/`
2. Retrain your model on the expanded dataset
3. Re-run steps 3–4 with the new weights
4. Each iteration: model improves → fewer corrections needed → faster review
