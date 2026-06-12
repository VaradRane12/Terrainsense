# TerrainSense — Multithreaded Refactor

## Data Labelling & Model Training
The code and scripts used for data labelling, dataset preprocessing, and model retraining can be found in two places:
1. Inside the `data_labelling_branch/` folder included directly in this directory.
2. On the dedicated `data_labelling` branch of the GitHub repository.

## File layout

```
terrainsense/
├── config.py          All constants and env-var tuning knobs
├── sensor.py          ToF 8×8 + IMU bridge (own daemon thread)
├── speech.py          espeak/espeak-ng queue worker (own daemon thread)
├── recording.py       Video writer + TTS narration mux (own daemon thread)
├── inference.py       TFLite model, preprocess, postprocess, OSD drawing
├── inference_fps.py   Rolling FPS tracker (tiny, avoids circular import)
├── camera.py          Picamera2 capture thread + inference thread + MJPEG generator
└── app.py             Flask routes only — thin orchestration layer
```

## Thread map

| Thread name        | Module          | Job                                        |
|--------------------|-----------------|-------------------------------------------|
| `sensor-bridge`    | sensor.py       | Read JSON lines from sensor subprocess    |
| `speech`           | speech.py       | Run espeak utterances one at a time       |
| `recording-writer` | recording.py    | Write annotated frames to disk            |
| `camera-capture`   | camera.py       | Picamera2 → raw_frame_queue               |
| `inference`        | camera.py       | raw_frame_queue → TFLite → result_frame_queue |
| Flask (main)       | app.py          | HTTP requests + MJPEG stream              |

## Queue flow

```
Picamera2
   │
   ▼ (drop if full)
raw_frame_queue  [maxsize=2]
   │
   ▼  inference thread
TFLite model
   │
   ├──► recording write_queue  [maxsize=60]  → disk
   ├──► speech._speech_queue   [maxsize=6]   → espeak
   │
   ▼ (drop if full)
result_frame_queue  [maxsize=2]
   │
   ▼  Flask thread
MJPEG stream → browser
```

## Startup

```bash
python app.py
```

Open `http://<pi-ip>:5000` in a browser.

## Environment variables (all optional)

| Variable                  | Default | Description                         |
|---------------------------|---------|-------------------------------------|
| `ENABLE_VOICE`            | `1`     | Set `0` to disable espeak           |
| `VOICE_INTERVAL_SEC`      | `3.0`   | Minimum seconds between utterances  |
| `VOICE_RATE`              | `165`   | espeak words-per-minute             |
| `ENABLE_SENSOR_BRIDGE`    | `1`     | Set `0` to disable ToF/IMU          |
| `SENSOR_STREAM_CMD`       | `python sensor_stream.py` | Command to start sensor subprocess |
| `TOF_ALERT_MM`            | `900`   | Distance threshold for ToF alerts   |
| `IGNORE_FAR_MM`           | `2200`  | Ignore detections beyond this range |
| `ENABLE_RECORD_AUDIO`     | `1`     | Bake TTS narration into recordings  |
| `PERSON_MOVE_PX_PER_SEC`  | `45.0`  | Motion threshold for person tracking|
| `FALL_ASPECT_THRESHOLD`   | `1.15`  | Width/height ratio for fall detect  |
| `WALK_CORRIDOR_WIDTH_RATIO`| `0.42` | Corridor width as fraction of frame |

## Dependencies

```
picamera2
opencv-python   (or opencv-python-headless)
numpy
flask
tflite-runtime  (or ai-edge-litert)
espeak / espeak-ng   (system package, for voice)
ffmpeg               (system package, for audio mux)
```
