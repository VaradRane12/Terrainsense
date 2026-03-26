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

# Stream performance tuning
STREAM_JPEG_QUALITY = int(os.environ.get("STREAM_JPEG_QUALITY", "50"))  # 50-95
INFERENCE_DROP_OLD_FRAMES = os.environ.get("INFERENCE_DROP_OLD_FRAMES", "1") == "1"
DRAW_TOF_GRID = os.environ.get("DRAW_TOF_GRID", "1") == "1"
STABILIZATION_ENABLED = os.environ.get("STABILIZATION_ENABLED", "0") == "1"
STABILIZATION_MAX_SHIFT_PX = float(os.environ.get("STABILIZATION_MAX_SHIFT_PX", "24.0"))

# ── Recording ──────────────────────────────────────────────────────────────
RECORDINGS_DIR         = "recordings"
DEFAULT_RECORD_MINUTES = 5.0
DEFAULT_RECORD_FPS     = 60.0
ENABLE_RECORD_AUDIO    = os.environ.get("ENABLE_RECORD_AUDIO", "1") == "1"
RECORD_AUDIO_RATE      = int(os.environ.get("RECORD_AUDIO_RATE", "16000"))

# ── Voice ──────────────────────────────────────────────────────────────────
ENABLE_VOICE       = os.environ.get("ENABLE_VOICE", "1") == "1"
VOICE_INTERVAL_SEC = float(os.environ.get("VOICE_INTERVAL_SEC", "1.0"))
VOICE_MIN_CONF     = float(os.environ.get("VOICE_MIN_CONF", "0.52"))
VOICE_STABLE_FRAMES = int(os.environ.get("VOICE_STABLE_FRAMES", "1"))
VOICE_RATE         = os.environ.get("VOICE_RATE", "165")
VOICE_TEST_TEXT    = os.environ.get("VOICE_TEST_TEXT", "Audio test from TerrainSense")

# Audio output switching
AUDIO_SWITCH_BT_CMD = os.environ.get(
    "AUDIO_SWITCH_BT_CMD",
    "pactl list short sinks | awk '/bluez_output/ {print $2; exit}' | xargs -I{} pactl set-default-sink {}",
)
AUDIO_SWITCH_AUX_CMD = os.environ.get(
    "AUDIO_SWITCH_AUX_CMD",
    "pactl list short sinks | awk '/analog-stereo|headphones|alsa_output/ {print $2; exit}' | xargs -I{} pactl set-default-sink {}",
)
AUDIO_QUERY_CMD = os.environ.get("AUDIO_QUERY_CMD", "pactl get-default-sink")
AUDIO_FORCE_AUX_ONLY = os.environ.get("AUDIO_FORCE_AUX_ONLY", "1") == "1"

# ── Sensor bridge ──────────────────────────────────────────────────────────
ENABLE_SENSOR_BRIDGE = os.environ.get("ENABLE_SENSOR_BRIDGE", "1") == "1"
SENSOR_STREAM_CMD    = os.environ.get("SENSOR_STREAM_CMD", "python sensor_stream.py")
TOF_ALERT_MM         = int(os.environ.get("TOF_ALERT_MM", "1200"))
IGNORE_FAR_MM        = int(os.environ.get("IGNORE_FAR_MM", "9000"))
TOF_PATH_MIN_VALID_CELLS = int(os.environ.get("TOF_PATH_MIN_VALID_CELLS", "2"))
TOF_PATH_CONFIRM_TOL_MM  = int(os.environ.get("TOF_PATH_CONFIRM_TOL_MM", "1400"))
TOF_SPEECH_REQUIRE_PATH  = os.environ.get("TOF_SPEECH_REQUIRE_PATH", "0") == "1"
TOF_ROTATE_180           = os.environ.get("TOF_ROTATE_180", "1") == "1"
TOF_FRONT_PERCENTILE     = float(os.environ.get("TOF_FRONT_PERCENTILE", "30"))

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
