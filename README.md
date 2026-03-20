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

### Core pipeline
```bash
python 1_extract_frames.py       # ~2 min  — pulls frames from all 3 clips
python 2_deduplicate.py          # ~3 min  — drops near-identical frames
python 3_run_inference.py        # ~5 min  — generates pseudo-labels + flags low-conf
python 4_prep_review.py          # ~1 min  — organises batches for labelImg
```

Then open labelImg and start with the priority batch:
```bash
labelImg review/priority/
```

### Utility & maintenance scripts
```bash
python bugfix_labelfix_roboflow.py   # fix Roboflow polygon exports → YOLO bounding boxes
python datasetcheck.py               # validate label format (4-value YOLO boxes)
python merge_dataset.py              # merge reviewed labels into main training data
python priority_rerun_tolabel-studio.py  # convert YOLO labels → Label Studio JSON
python removing_samplebatch.py       # remove first sample-batch frames before retraining
```

### Retraining
```bash
# After merging reviewed data:
python train.py --weights weights/your_model.pt --data data.yaml
```

---

## Script reference

### Core pipeline

| # | Script | Input | Output | Notes |
|---|--------|-------|--------|-------|
| 1 | `1_extract_frames.py` | `clip*.mp4` | `data/raw_frames/` | Samples at `INTERVAL_SEC` |
| 2 | `2_deduplicate.py` | `data/raw_frames/` | `data/deduped_frames/` | Perceptual hash dedup |
| 3 | `3_run_inference.py` | `data/deduped_frames/` | `data/pseudo_labels/`, `priority_review.txt` | Flags low-confidence frames |
| 4 | `4_prep_review.py` | `priority_review.txt` | `review/priority/`, `review/normal/` | Organises batches for labelImg |

### Utility scripts

| Script | Purpose |
|--------|---------|
| `bugfix_labelfix_roboflow.py` | Converts Roboflow polygon-format labels (which export with more than 4 values per line) into standard YOLO bounding-box format (`class cx cy w h`). Run this on any Roboflow export before merging into the dataset. |
| `datasetcheck.py` | Validates every `.txt` label file to confirm each line has exactly 4 coordinate values (standard YOLO box format). Prints a report of malformed files so you can fix them before training. |
| `merge_dataset.py` | Copies manually reviewed and corrected frames + labels from the review folders into `data/images/train/` and `data/labels/train/`. Run after finishing a review batch and before retraining. |
| `priority_rerun_tolabel-studio.py` | Converts standard YOLO `.txt` labels into the JSON format expected by Label Studio. Use this when you want to do a review pass in Label Studio instead of labelImg (e.g., for polygon or keypoint workflows). |
| `removing_samplebatch.py` | Removes the frames and labels from the initial sample batch (used for the first retraining run) so they don't duplicate or skew subsequent prediction and labelling rounds. Run once, after the first retraining cycle is complete. |

---

## Retraining loop

```
1. Review frames in labelImg (priority/ first, then spot-check normal/)
2. Run merge_dataset.py  →  reviewed data lands in data/images/train/ + data/labels/train/
3. Retrain on combined old + new data
4. Re-run steps 3–4 with the new weights
5. Repeat — each iteration the model improves and fewer corrections are needed
```

> **First run only:** after the initial sample-batch retraining, run `removing_samplebatch.py`
> to prevent those frames from appearing again in future prediction/labelling rounds.

---

## Tuning notes

| Parameter | File | Default | Change if... |
|-----------|------|---------|-------------|
| `INTERVAL_SEC` | step 1 | 0.1 s | Action is fast → lower; clip is slow → higher |
| `HASH_THRESH` | step 2 | 8 | Too many dupes kept → raise; too many dropped → lower |
| `CONF_THRESHOLD` | step 3 | 0.25 | Missing detections → lower; too much noise → raise |
| `FLAG_THRESHOLD` | step 3 | 0.50 | Controls how many frames get priority review |