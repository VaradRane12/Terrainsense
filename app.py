from flask import Flask, Response, jsonify, request
import cv2
import numpy as np
import time
import os
import threading
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

record_lock = threading.Lock()
record_state = {
    "active": False,
    "writer": None,
    "path": None,
    "fps": DEFAULT_RECORD_FPS,
    "started_at": 0.0,
    "stop_at": 0.0,
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

# ───────────────────────────────────────────────
#  POSTPROCESS
# ───────────────────────────────────────────────
def postprocess(output, frame):
    h_frame, w_frame = frame.shape[:2]
    preds        = output[0].squeeze(0).T
    class_scores = sigmoid(preds[:, 4:])
    confidences  = np.max(class_scores, axis=1)
    class_ids    = np.argmax(class_scores, axis=1)
    mask = confidences > CONF_THRESH
    if not np.any(mask):
        return None, None, 0.0
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
        return None, None, 0.0
    best_i       = max(indices.flatten(), key=lambda i: confs_f[i])
    x1, y1, x2, y2 = boxes_xyxy[best_i]
    conf         = float(confs_f[best_i])
    label, color = CLASS_LABELS.get(int(classes_f[best_i]), ("UNKNOWN", (255,255,255)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    text        = f"{label}  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1-th-12), (x1+tw+8, y1), color, -1)
    cv2.putText(frame, text, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    return label, color, conf


def create_frame_source():
    if not USE_PI_CAMERA:
        raise RuntimeError("USE_PI_CAMERA is set to 0, but this app now requires Picamera2 camera input.")

    if Picamera2 is None:
        raise RuntimeError("Picamera2 is not installed. Install it and restart the app.")

    print("[INFO] Starting Picamera2 stream...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.2)
    return "picamera2", picam2


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
    _, source = create_frame_source()
    fps_list = []
    try:
        while True:
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
            label, color, conf = result if result[0] else ("ALL CLEAR", (0,255,0), 0.0)
            cv2.rectangle(frame, (0, 0), (w, 60), (0,0,0), -1)
            cv2.putText(frame, f"  {label}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
            cv2.putText(frame, f"Conf: {conf:.2f}  |  FPS: {avg_fps:.1f}", (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            maybe_write_recording(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        with record_lock:
            if record_state["active"]:
                _stop_recording_locked()
        source.stop()

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
        "<div id='recStatus' style='margin-top:8px;font-size:14px;'></div>"
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

if __name__ == '__main__':
    print("[INFO] Open browser at http://<raspberry-pi-ip>:5000")
    print("[INFO] Camera source: Picamera2 only")
    app.run(host='0.0.0.0', port=5000, debug=False)