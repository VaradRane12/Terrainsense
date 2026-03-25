import os

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "model.tflite"
INPUT_SIZE  = 320
CONF_THRESH = 0.50
NMS_THRESH  = 0.30

# ── Model optimization (Pi 4) ──────────────────────────────────────────────
# Set to "coral" if you have Edge TPU (3-5x speedup), "nnapi" for Android, "gpu" for Mali
MODEL_DELEGATE = os.environ.get("MODEL_DELEGATE", "auto")  # "auto", "cpu", "coral", "gpu", "nnapi"

# Input resolution optimization: scale down for faster inference on Pi 4
# If camera is 640x480 but model is 320x320, we'll crop to square first
# Reducing resolution by scale_factor speeds up inference
MODEL_INPUT_SCALE = float(os.environ.get("MODEL_INPUT_SCALE", "1.0"))  # 0.75 = 25% faster, slightly lower accuracy

# Thread count for TFLite inference (Pi 4 = 4 cores, but keep 1 for OS)
MODEL_NUM_THREADS = int(os.environ.get("MODEL_NUM_THREADS", "3"))  # Leave 1 core for system

# ── Camera ─────────────────────────────────────────────────────────────────
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

# Capture FPS limit on Pi 4 (sensor default is 30, can reduce to save CPU)
CAMERA_FPS = int(os.environ.get("CAMERA_FPS", "30"))

# ── Recording ──────────────────────────────────────────────────────────────
RECORDINGS_DIR         = "recordings"
DEFAULT_RECORD_MINUTES = 5.0
DEFAULT_RECORD_FPS     = 20.0
ENABLE_RECORD_AUDIO    = os.environ.get("ENABLE_RECORD_AUDIO", "1") == "1"
RECORD_AUDIO_RATE      = int(os.environ.get("RECORD_AUDIO_RATE", "16000"))

# ── Voice ──────────────────────────────────────────────────────────────────
ENABLE_VOICE       = os.environ.get("ENABLE_VOICE", "1") == "1"
VOICE_INTERVAL_SEC = float(os.environ.get("VOICE_INTERVAL_SEC", "3.0"))
VOICE_RATE         = os.environ.get("VOICE_RATE", "165")
VOICE_TEST_TEXT    = os.environ.get("VOICE_TEST_TEXT", "Audio test from TerrainSense")

# ── Sensor bridge ──────────────────────────────────────────────────────────
ENABLE_SENSOR_BRIDGE = os.environ.get("ENABLE_SENSOR_BRIDGE", "1") == "1"
SENSOR_STREAM_CMD    = os.environ.get("SENSOR_STREAM_CMD", "python sensor_stream.py")
TOF_ALERT_MM         = int(os.environ.get("TOF_ALERT_MM", "900"))
IGNORE_FAR_MM        = int(os.environ.get("IGNORE_FAR_MM", "2200"))

# ── Person / fall detection ────────────────────────────────────────────────
PERSON_MOVE_PX_PER_SEC  = float(os.environ.get("PERSON_MOVE_PX_PER_SEC", "45.0"))
FALL_ASPECT_THRESHOLD   = float(os.environ.get("FALL_ASPECT_THRESHOLD", "1.15"))
FALL_HEIGHT_DROP_RATIO  = float(os.environ.get("FALL_HEIGHT_DROP_RATIO", "0.65"))

# ── Walking corridor ───────────────────────────────────────────────────────
WALK_CORRIDOR_WIDTH_RATIO = float(os.environ.get("WALK_CORRIDOR_WIDTH_RATIO", "0.42"))
YAW_SHIFT_SCALE           = float(os.environ.get("YAW_SHIFT_SCALE", "0.55"))

# ── Detection class labels ─────────────────────────────────────────────────
CLASS_LABELS = {
    0: ("OBSTACLE", (0, 165, 255)),
    1: ("PERSON",   (255, 0, 0)),
    2: ("POTHOLE",  (0, 0, 255)),
    3: ("VEHICLE",  (0, 255, 255)),
}
