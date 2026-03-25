from flask import Flask, Response, jsonify, request
import cv2
import numpy as np
import time
import os
import threading
import queue
import subprocess
import json
import math
from datetime import datetime

INTERPRETER_BACKEND = None
try:
    from tflite_runtime.interpreter import Interpreter
    INTERPRETER_BACKEND = "tflite_runtime"
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
        INTERPRETER_BACKEND = "ai_edge_litert"
    except ImportError as exc:
        raise ImportError(
            "No TFLite interpreter found. Install tflite-runtime or ai-edge-litert."
        ) from exc

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

app = Flask(__name__)

MODEL_PATH  = "model.tflite"
INPUT_SIZE  = 320
CONF_THRESH = 0.50
NMS_THRESH  = 0.30
USE_PI_CAMERA = os.environ.get("USE_PI_CAMERA", "1") == "1"
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
RECORDINGS_DIR = "recordings"
DEFAULT_RECORD_MINUTES = 5.0
DEFAULT_RECORD_FPS = 20.0
ENABLE_VOICE = os.environ.get("ENABLE_VOICE", "1") == "1"
VOICE_INTERVAL_SEC = float(os.environ.get("VOICE_INTERVAL_SEC", "3.0"))
VOICE_RATE = os.environ.get("VOICE_RATE", "165")
VOICE_TEST_TEXT = os.environ.get("VOICE_TEST_TEXT", "Audio test from TerrainSense")
ENABLE_SENSOR_BRIDGE = os.environ.get("ENABLE_SENSOR_BRIDGE", "1") == "1"
SENSOR_STREAM_CMD = os.environ.get("SENSOR_STREAM_CMD", "python sensor_stream.py")
TOF_ALERT_MM = int(os.environ.get("TOF_ALERT_MM", "900"))
PERSON_MOVE_PX_PER_SEC = float(os.environ.get("PERSON_MOVE_PX_PER_SEC", "45.0"))
FALL_ASPECT_THRESHOLD = float(os.environ.get("FALL_ASPECT_THRESHOLD", "1.15"))
FALL_HEIGHT_DROP_RATIO = float(os.environ.get("FALL_HEIGHT_DROP_RATIO", "0.65"))

record_lock = threading.Lock()
record_state = {
    "active": False,
    "writer": None,
    "path": None,
    "fps": DEFAULT_RECORD_FPS,
    "started_at": 0.0,
    "stop_at": 0.0,
}

speech_queue = queue.Queue(maxsize=6)
speech_state = {
    "enabled": ENABLE_VOICE,
    "last_text": "",
    "last_time": 0.0,
    "warned_missing_engine": False,
}

camera_lock = threading.Lock()
capture_lock = threading.Lock()
camera_state = {
    "instance": None,
    "users": 0,
}

sensor_lock = threading.Lock()
sensor_state = {
    "enabled": ENABLE_SENSOR_BRIDGE,
    "front_mm": None,
    "pitch_deg": 0.0,
    "quat": [1.0, 0.0, 0.0, 0.0],
    "distances": [0] * 64,
    "status": [0] * 64,
    "last_update": 0.0,
    "error": None,
}

person_state = {
    "last_center_x": None,
    "last_height": None,
    "last_time": 0.0,
}

CLASS_LABELS = {
    0: ("OBSTACLE", (0, 165, 255)),
    1: ("PERSON",   (255, 0, 0)),
    2: ("POTHOLE",  (0, 0, 255)),
    3: ("VEHICLE",  (0, 255, 255)),
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ───────────────────────────────────────────────
#  LOAD TFLITE MODEL
# ───────────────────────────────────────────────
print("[INFO] Loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print(f"[INFO] Interpreter backend: {INTERPRETER_BACKEND}")

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype    = input_details[0]['dtype']
print(f"[INFO] Model input dtype: {input_dtype}")
print("[INFO] ✅ TFLite model loaded!")

# ───────────────────────────────────────────────
#  PREPROCESS
# ───────────────────────────────────────────────
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)              # HWC → CHW
    img = np.expand_dims(img, axis=0)         # add batch dim

    # Handle quantized models
    if input_dtype == np.int8:
        scale, zero_point = input_details[0]['quantization']
        img = (img / scale + zero_point).astype(np.int8)
    elif input_dtype == np.uint8:
        img = (img * 255).astype(np.uint8)

    return img

# ───────────────────────────────────────────────
#  RUN TFLITE INFERENCE
# ───────────────────────────────────────────────
def run_inference(frame):
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize if INT8
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    return [output]


def _extract_front_mm(distances, status):
    # Center 4x4 cells in 8x8 grid.
    idxs = [
        18, 19, 20, 21,
        26, 27, 28, 29,
        34, 35, 36, 37,
        42, 43, 44, 45,
    ]

    vals = [int(distances[i]) for i in idxs if i < len(distances) and i < len(status) and status[i] == 5 and distances[i] > 0]
    if not vals:
        # Fallback when status bits are sparse but distances exist.
        vals = [int(distances[i]) for i in idxs if i < len(distances) and distances[i] > 0]
    if not vals:
        return None
    return int(np.median(vals))


def _quat_to_pitch_deg(quat):
    if not quat or len(quat) != 4:
        return 0.0
    w, x, y, z = [float(v) for v in quat]
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    return float(math.degrees(math.asin(sinp)))


def _sensor_stream_worker():
    if not ENABLE_SENSOR_BRIDGE:
        with sensor_lock:
            sensor_state["error"] = "Sensor bridge disabled"
        return

    try:
        proc = subprocess.Popen(
            SENSOR_STREAM_CMD,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        with sensor_lock:
            sensor_state["error"] = f"Failed to start sensor stream: {exc}"
        return

    print(f"[INFO] Sensor bridge started: {SENSOR_STREAM_CMD}")

    if proc.stdout is None:
        with sensor_lock:
            sensor_state["error"] = "Sensor stream stdout unavailable"
        return

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            pkt = json.loads(line)
            distances = list(pkt.get("distances", []))[:64]
            status = list(pkt.get("status", []))[:64]
            quat = list(pkt.get("quat", [1.0, 0.0, 0.0, 0.0]))[:4]

            if len(distances) < 64:
                distances += [0] * (64 - len(distances))
            if len(status) < 64:
                status += [0] * (64 - len(status))
            if len(quat) < 4:
                quat = [1.0, 0.0, 0.0, 0.0]

            front_mm = _extract_front_mm(distances, status)
            pitch_deg = _quat_to_pitch_deg(quat)

            with sensor_lock:
                sensor_state["front_mm"] = front_mm
                sensor_state["pitch_deg"] = pitch_deg
                sensor_state["quat"] = quat
                sensor_state["distances"] = distances
                sensor_state["status"] = status
                sensor_state["last_update"] = time.time()
                sensor_state["error"] = None
        except Exception as exc:
            with sensor_lock:
                sensor_state["error"] = f"Sensor parse error: {exc}"

    with sensor_lock:
        sensor_state["error"] = "Sensor stream stopped"


def get_sensor_snapshot():
    with sensor_lock:
        return dict(sensor_state)


def _tof_cell_color(d_mm, st):
    if st != 5 or d_mm <= 0:
        return (45, 45, 45)
    d = max(200, min(3000, int(d_mm)))
    t = (d - 200) / 2800.0
    return (0, int(255 * t), int(255 * (1.0 - t)))


def draw_tof_grid(frame, snap):
    dists = snap.get("distances", [0] * 64)
    stats = snap.get("status", [0] * 64)

    h, w = frame.shape[:2]
    cell = 14
    gap = 2
    size = (cell + gap) * 8 + gap
    x0 = w - size - 18
    y0 = 70

    cv2.rectangle(frame, (x0 - 8, y0 - 24), (x0 + size + 8, y0 + size + 8), (20, 20, 20), -1)
    cv2.putText(frame, "ToF 8x8", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    for r in range(8):
        for c in range(8):
            i = r * 8 + c
            col = _tof_cell_color(dists[i], stats[i])
            x1 = x0 + gap + c * (cell + gap)
            y1 = y0 + gap + r * (cell + gap)
            x2 = x1 + cell
            y2 = y1 + cell
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), 1)

    # Center 4x4 region.
    cx1 = x0 + gap + 2 * (cell + gap)
    cy1 = y0 + gap + 2 * (cell + gap)
    cx2 = x0 + gap + 6 * (cell + gap) - gap
    cy2 = y0 + gap + 6 * (cell + gap) - gap
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 1)


def _tof_mm_for_bbox(snap, x1, x2, frame_w):
    dists = snap.get("distances", [0] * 64)
    stats = snap.get("status", [0] * 64)

    c0 = max(0, min(7, int((x1 / max(1, frame_w)) * 8)))
    c1 = max(0, min(7, int((x2 / max(1, frame_w)) * 8)))
    if c1 < c0:
        c0, c1 = c1, c0

    vals = []
    for r in range(8):
        for c in range(c0, c1 + 1):
            i = r * 8 + c
            if i < len(dists) and i < len(stats) and stats[i] == 5 and dists[i] > 0:
                vals.append(int(dists[i]))

    if not vals:
        return snap.get("front_mm")
    return int(np.median(vals))


def _tof_nearest_side(snap):
    dists = snap.get("distances", [0] * 64)
    stats = snap.get("status", [0] * 64)
    bands = {
        "left": [0, 1, 2],
        "center": [3, 4],
        "right": [5, 6, 7],
    }

    best_side = None
    best_mm = None
    for side, cols in bands.items():
        vals = []
        for r in range(8):
            for c in cols:
                i = r * 8 + c
                if i < len(dists) and i < len(stats) and stats[i] == 5 and dists[i] > 0:
                    vals.append(int(dists[i]))
        if vals:
            mm = int(np.median(vals))
            if best_mm is None or mm < best_mm:
                best_mm = mm
                best_side = side
    return best_side, best_mm


def _person_motion_and_fall(x1, y1, x2, y2, frame_w, frame_h, snap, now):
    center_x = 0.5 * (x1 + x2)
    height = max(1.0, float(y2 - y1))
    width = max(1.0, float(x2 - x1))

    if center_x < frame_w * 0.38:
        person_side = "left"
    elif center_x > frame_w * 0.62:
        person_side = "right"
    else:
        person_side = "center"

    motion = "steady"
    prev_x = person_state["last_center_x"]
    prev_t = person_state["last_time"]
    if prev_x is not None and prev_t > 0.0:
        dt = max(1e-3, now - prev_t)
        vx = (center_x - prev_x) / dt
        if vx > PERSON_MOVE_PX_PER_SEC:
            motion = "moving right"
        elif vx < -PERSON_MOVE_PX_PER_SEC:
            motion = "moving left"

    aspect = width / height
    prev_h = person_state["last_height"]
    drop = (prev_h is not None and height < float(prev_h) * FALL_HEIGHT_DROP_RATIO)
    tilt = abs(float(snap.get("pitch_deg", 0.0)))
    fall_risk = (aspect > FALL_ASPECT_THRESHOLD and height < frame_h * 0.45) or (drop and tilt > 8.0)

    person_mm = _tof_mm_for_bbox(snap, x1, x2, frame_w)
    tof_side, _ = _tof_nearest_side(snap)

    person_state["last_center_x"] = center_x
    person_state["last_height"] = height
    person_state["last_time"] = now

    return {
        "side": person_side,
        "motion": motion,
        "fall_risk": fall_risk,
        "person_mm": person_mm,
        "tof_side": tof_side,
    }


def direction_guidance(label, x1, x2, width, snap=None):
    center_x = (x1 + x2) / 2.0
    left_band = width * 0.38
    right_band = width * 0.62

    if center_x < left_band:
        side = "left"
        move = "right"
    elif center_x > right_band:
        side = "right"
        move = "left"
    else:
        side = "center"
        move = "stop"

    if move == "stop":
        guidance_text = f"{label} ahead - slow/stop"
        speech_text = f"Stop. {label.lower()} ahead"
    else:
        guidance_text = f"{label} on {side} - move {move}"
        speech_text = f"Move {move}. {label.lower()} ahead"

    if snap is not None and snap.get("front_mm") is not None and snap["front_mm"] < TOF_ALERT_MM:
        guidance_text = f"{guidance_text} | {snap['front_mm']} mm"
        if move == "stop":
            speech_text = f"Stop. {label.lower()} at {snap['front_mm']} millimeters"

    return guidance_text, speech_text


def _speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            speech_queue.task_done()
            break

        if speech_state["enabled"]:
            ok = False
            for engine in ("espeak", "espeak-ng"):
                try:
                    subprocess.run(
                        [engine, "-s", VOICE_RATE, text],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    ok = True
                    break
                except FileNotFoundError:
                    continue

            if not ok:
                if not speech_state["warned_missing_engine"]:
                    print("[WARN] No speech engine found. Install espeak or espeak-ng to enable voice guidance.")
                    speech_state["warned_missing_engine"] = True
                speech_state["enabled"] = False

        speech_queue.task_done()


def queue_speech(text):
    if not text or not speech_state["enabled"]:
        return

    now = time.time()
    if now - speech_state["last_time"] < VOICE_INTERVAL_SEC:
        return

    if text == speech_state["last_text"] and now - speech_state["last_time"] < (VOICE_INTERVAL_SEC * 2.0):
        return

    speech_state["last_text"] = text
    speech_state["last_time"] = now

    try:
        speech_queue.put_nowait(text)
    except queue.Full:
        pass


def set_voice_enabled(enabled):
    speech_state["enabled"] = bool(enabled)
    return {
        "ok": True,
        "voice_enabled": speech_state["enabled"],
        "interval_sec": VOICE_INTERVAL_SEC,
    }


def get_voice_status():
    return {
        "ok": True,
        "voice_enabled": speech_state["enabled"],
        "interval_sec": VOICE_INTERVAL_SEC,
    }


speech_thread = threading.Thread(target=_speech_worker, daemon=True)
speech_thread.start()

# ───────────────────────────────────────────────
#  POSTPROCESS
# ───────────────────────────────────────────────
def postprocess(output, frame):
    snap = get_sensor_snapshot()
    now = time.time()
    h_frame, w_frame = frame.shape[:2]
    preds        = output[0].squeeze(0).T
    class_scores = sigmoid(preds[:, 4:])
    confidences  = np.max(class_scores, axis=1)
    class_ids    = np.argmax(class_scores, axis=1)
    mask = confidences > CONF_THRESH
    if not np.any(mask):
        if snap.get("front_mm") is not None and snap["front_mm"] < TOF_ALERT_MM:
            return None, (0, 165, 255), 0.0, f"TOF obstacle {snap['front_mm']} mm", f"Stop. obstacle at {snap['front_mm']} millimeters"
        return None, (0, 255, 0), 0.0, "Path clear", None
    boxes_f   = preds[:, :4][mask]
    confs_f   = confidences[mask].tolist()
    classes_f = class_ids[mask]
    scale_x = w_frame / INPUT_SIZE
    scale_y = h_frame / INPUT_SIZE
    boxes_xyxy = []
    for box in boxes_f:
        cx, cy, bw, bh = box
        x1 = max(0, int((cx - bw/2) * scale_x))
        y1 = max(0, int((cy - bh/2) * scale_y))
        x2 = min(w_frame, int((cx + bw/2) * scale_x))
        y2 = min(h_frame, int((cy + bh/2) * scale_y))
        boxes_xyxy.append([x1, y1, x2, y2])
    boxes_nms = [[x1, y1, x2-x1, y2-y1] for x1,y1,x2,y2 in boxes_xyxy]
    indices   = cv2.dnn.NMSBoxes(boxes_nms, confs_f, CONF_THRESH, NMS_THRESH)
    if len(indices) == 0:
        if snap.get("front_mm") is not None and snap["front_mm"] < TOF_ALERT_MM:
            return None, (0, 165, 255), 0.0, f"TOF obstacle {snap['front_mm']} mm", f"Stop. obstacle at {snap['front_mm']} millimeters"
        return None, (0, 255, 0), 0.0, "Path clear", None
    best_i       = max(indices.flatten(), key=lambda i: confs_f[i])
    x1, y1, x2, y2 = boxes_xyxy[best_i]
    conf         = float(confs_f[best_i])
    label, color = CLASS_LABELS.get(int(classes_f[best_i]), ("UNKNOWN", (255,255,255)))
    guidance_text, speech_text = direction_guidance(label, x1, x2, w_frame, snap=snap)

    if label == "PERSON":
        person_info = _person_motion_and_fall(x1, y1, x2, y2, w_frame, h_frame, snap, now)
        if person_info["fall_risk"]:
            guidance_text = "PERSON fall risk - assist immediately"
            speech_text = "Alert. person may be falling"
            color = (0, 0, 255)
        else:
            mm = person_info["person_mm"]
            if mm is not None:
                guidance_text = f"PERSON {person_info['motion']} | {mm} mm"
                speech_text = f"Person {person_info['motion']}. Distance {mm} millimeters"
            else:
                guidance_text = f"PERSON {person_info['motion']}"
                speech_text = f"Person {person_info['motion']}"

            if person_info["tof_side"] and person_info["tof_side"] != person_info["side"]:
                guidance_text = f"{guidance_text} | ToF nearer {person_info['tof_side']}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    text        = f"{label}  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1-th-12), (x1+tw+8, y1), color, -1)
    cv2.putText(frame, text, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    return label, color, conf, guidance_text, speech_text


def acquire_camera():
    if not USE_PI_CAMERA:
        raise RuntimeError("USE_PI_CAMERA is set to 0, but this app now requires Picamera2 camera input.")

    if Picamera2 is None:
        raise RuntimeError("Picamera2 is not installed. Install it and restart the app.")

    with camera_lock:
        if camera_state["instance"] is None:
            print("[INFO] Starting Picamera2 stream...")
            picam2 = Picamera2()
            try:
                config = picam2.create_video_configuration(
                    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
                )
                picam2.configure(config)
                picam2.start()
                time.sleep(0.2)
            except Exception:
                try:
                    picam2.stop()
                except Exception:
                    pass
                try:
                    picam2.close()
                except Exception:
                    pass
                raise
            camera_state["instance"] = picam2

        camera_state["users"] += 1
        return camera_state["instance"]


def release_camera():
    with camera_lock:
        if camera_state["users"] > 0:
            camera_state["users"] -= 1

        if camera_state["users"] == 0 and camera_state["instance"] is not None:
            picam2 = camera_state["instance"]
            camera_state["instance"] = None
            try:
                picam2.stop()
            except Exception:
                pass
            try:
                picam2.close()
            except Exception:
                pass


def _create_recording_path():
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RECORDINGS_DIR, f"terrain_{stamp}.mp4")


def _stop_recording_locked():
    writer = record_state["writer"]
    if writer is not None:
        writer.release()

    was_active = record_state["active"]
    saved_path = record_state["path"]
    started_at = record_state["started_at"]
    duration_sec = max(0.0, time.time() - started_at) if started_at else 0.0

    record_state["active"] = False
    record_state["writer"] = None
    record_state["path"] = None
    record_state["started_at"] = 0.0
    record_state["stop_at"] = 0.0

    return {
        "ok": True,
        "was_active": was_active,
        "saved_to": saved_path,
        "duration_sec": round(duration_sec, 2),
    }


def start_recording(minutes, fps):
    with record_lock:
        if record_state["active"]:
            return {
                "ok": False,
                "error": "Recording already active",
                "saved_to": record_state["path"],
            }

        if minutes <= 0:
            raise ValueError("minutes must be > 0")
        if fps <= 0:
            raise ValueError("fps must be > 0")

        now = time.time()
        record_state["active"] = True
        record_state["writer"] = None
        record_state["path"] = _create_recording_path()
        record_state["fps"] = float(fps)
        record_state["started_at"] = now
        record_state["stop_at"] = now + (float(minutes) * 60.0)

        return {
            "ok": True,
            "recording": True,
            "minutes": float(minutes),
            "fps": float(fps),
            "saved_to": record_state["path"],
        }


def stop_recording():
    with record_lock:
        return _stop_recording_locked()


def get_record_status():
    with record_lock:
        left = max(0.0, record_state["stop_at"] - time.time()) if record_state["active"] else 0.0
        return {
            "ok": True,
            "recording": record_state["active"],
            "saved_to": record_state["path"],
            "seconds_left": round(left, 1),
            "fps": record_state["fps"],
        }


def maybe_write_recording(frame):
    with record_lock:
        if not record_state["active"]:
            return

        if time.time() >= record_state["stop_at"]:
            _stop_recording_locked()
            return

        if record_state["writer"] is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(record_state["path"], fourcc, record_state["fps"], (w, h))
            if not writer.isOpened():
                _stop_recording_locked()
                raise RuntimeError("Failed to open output video file for recording")
            record_state["writer"] = writer

        record_state["writer"].write(frame)

# ───────────────────────────────────────────────
#  FRAME GENERATOR
# ───────────────────────────────────────────────
def generate_frames():
    source = acquire_camera()
    fps_list = []
    try:
        while True:
            with capture_lock:
                frame = source.capture_array()
            if frame is None:
                continue
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            h, w, _ = frame.shape
            t1     = time.time()
            output = run_inference(frame)
            t2     = time.time()
            fps = 1.0 / (t2 - t1 + 1e-6)
            fps_list.append(fps)
            avg_fps = sum(fps_list[-10:]) / len(fps_list[-10:])
            result = postprocess(output, frame)
            label, color, conf, guidance, speech_text = result
            display_label = label if label else "ALL CLEAR"
            cv2.rectangle(frame, (0, 0), (w, 60), (0,0,0), -1)
            cv2.putText(frame, f"  {display_label}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
            cv2.putText(frame, guidance, (20, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80,255,200), 2)
            cv2.putText(frame, f"Conf: {conf:.2f}  |  FPS: {avg_fps:.1f}", (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            snap = get_sensor_snapshot()
            if not snap.get("enabled", False):
                tof_text = "ToF front: disabled"
            elif snap.get("front_mm") is not None:
                tof_text = f"ToF front: {snap['front_mm']} mm"
            elif snap.get("error"):
                tof_text = "ToF front: sensor error"
            else:
                tof_text = "ToF front: no valid target"
            cv2.putText(frame, tof_text, (w - 265, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {snap.get('pitch_deg', 0.0):.1f} deg", (w - 265, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            draw_tof_grid(frame, snap)
            queue_speech(speech_text)
            maybe_write_recording(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        with record_lock:
            if record_state["active"]:
                _stop_recording_locked()
        release_camera()

# ───────────────────────────────────────────────
#  FLASK ROUTES
# ───────────────────────────────────────────────
@app.route('/')
def index():
    return (
        "<html><body style='background:#111;text-align:center;margin:0;padding:20px;'>"
        "<h1 style='color:#00ff99;font-family:Arial;'>TerrainSense Live</h1>"
        "<div style='margin-bottom:12px;color:#d5ffe8;font-family:Arial;'>"
        "Minutes: <input id='mins' type='number' min='1' value='5' style='width:70px;'>"
        "<button onclick='startRec()' style='margin-left:8px;padding:8px 12px;'>Start Recording</button>"
        "<button onclick='stopRec()' style='margin-left:8px;padding:8px 12px;'>Stop</button>"
        "<button onclick='voiceOn()' style='margin-left:8px;padding:8px 12px;'>Voice ON</button>"
        "<button onclick='voiceOff()' style='margin-left:8px;padding:8px 12px;'>Voice OFF</button>"
        "<button onclick='voiceTest()' style='margin-left:8px;padding:8px 12px;'>Voice TEST</button>"
        "<div id='recStatus' style='margin-top:8px;font-size:14px;'></div>"
        "<div id='voiceStatus' style='margin-top:4px;font-size:14px;'></div>"
        "</div>"
        "<img src='/video_feed' style='width:100%;max-width:860px;border:2px solid #00ff99;border-radius:8px;'>"
        "<script>"
        "async function refreshStatus(){"
        "  const r = await fetch('/record/status');"
        "  const j = await r.json();"
        "  const status = j.recording"
        "    ? `Recording ON | ${j.seconds_left}s left | File: ${j.saved_to}`"
        "    : 'Recording OFF';"
        "  document.getElementById('recStatus').textContent = status;"
        "  const vr = await fetch('/voice/status');"
        "  const vj = await vr.json();"
        "  document.getElementById('voiceStatus').textContent = vj.voice_enabled ? 'Voice ON' : 'Voice OFF';"
        "}"
        "async function startRec(){"
        "  const m = document.getElementById('mins').value || '5';"
        "  const r = await fetch(`/record/start?minutes=${encodeURIComponent(m)}`);"
        "  const j = await r.json();"
        "  document.getElementById('recStatus').textContent = j.error || ('Started: ' + j.saved_to);"
        "}"
        "async function stopRec(){"
        "  const r = await fetch('/record/stop');"
        "  const j = await r.json();"
        "  document.getElementById('recStatus').textContent = j.saved_to ? ('Saved: ' + j.saved_to) : 'Stopped';"
        "}"
        "async function voiceOn(){"
        "  await fetch('/voice/on');"
        "  refreshStatus();"
        "}"
        "async function voiceOff(){"
        "  await fetch('/voice/off');"
        "  refreshStatus();"
        "}"
        "async function voiceTest(){"
        "  const r = await fetch('/voice/test');"
        "  const j = await r.json();"
        "  document.getElementById('voiceStatus').textContent = j.message || j.error || 'Voice test sent';"
        "}"
        "setInterval(refreshStatus, 1000);"
        "refreshStatus();"
        "</script>"
        "</body></html>"
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/record/start', methods=['GET', 'POST'])
def record_start():
    try:
        minutes = float(request.args.get('minutes', DEFAULT_RECORD_MINUTES))
        fps = float(request.args.get('fps', DEFAULT_RECORD_FPS))
        result = start_recording(minutes=minutes, fps=fps)
        status_code = 200 if result.get("ok", False) else 409
        return jsonify(result), status_code
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route('/record/stop', methods=['GET', 'POST'])
def record_stop():
    return jsonify(stop_recording()), 200


@app.route('/record/status', methods=['GET'])
def record_status():
    return jsonify(get_record_status()), 200


@app.route('/voice/status', methods=['GET'])
def voice_status():
    return jsonify(get_voice_status()), 200


@app.route('/voice/on', methods=['GET', 'POST'])
def voice_on():
    return jsonify(set_voice_enabled(True)), 200


@app.route('/voice/off', methods=['GET', 'POST'])
def voice_off():
    return jsonify(set_voice_enabled(False)), 200


@app.route('/voice/test', methods=['GET', 'POST'])
def voice_test():
    if not speech_state["enabled"]:
        return jsonify({"ok": False, "error": "Voice is OFF. Turn voice on first."}), 409

    queue_speech(VOICE_TEST_TEXT)
    return jsonify({"ok": True, "message": f"Queued voice test: {VOICE_TEST_TEXT}"}), 200


@app.route('/sensor/status', methods=['GET'])
def sensor_status():
    return jsonify({"ok": True, "sensor": get_sensor_snapshot()}), 200

if __name__ == '__main__':
    if ENABLE_SENSOR_BRIDGE:
        sensor_thread = threading.Thread(target=_sensor_stream_worker, daemon=True)
        sensor_thread.start()
        print(f"[INFO] Sensor bridge: enabled ({SENSOR_STREAM_CMD})")
    else:
        with sensor_lock:
            sensor_state["error"] = "Sensor bridge disabled"
        print("[INFO] Sensor bridge: disabled")
    print("[INFO] Open browser at http://<raspberry-pi-ip>:5000")
    print("[INFO] Camera source: Picamera2 only")
    if ENABLE_VOICE:
        print(f"[INFO] Voice guidance: enabled (min interval {VOICE_INTERVAL_SEC:.1f}s)")
    else:
        print("[INFO] Voice guidance: disabled")
    app.run(host='0.0.0.0', port=5000, debug=False)