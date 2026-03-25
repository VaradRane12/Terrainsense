import os

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "model.tflite"
INPUT_SIZE  = 320
CONF_THRESH = 0.50
NMS_THRESH  = 0.30

# ── Camera ─────────────────────────────────────────────────────────────────
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

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
