import cv2
import numpy as np
import time
import threading
from tflite_runtime.interpreter import Interpreter
from flask import Flask, Response, jsonify

# ── CONFIG ─────────────────────────────────────
MODEL_PATH   = "best_int8.tflite"
VIDEO_PATH   = 0
CONF_THRESH  = 0.5
TOP_K        = 10
NUM_THREADS  = 4
JPEG_QUALITY = 55
FRAME_SCALE  = 0.5

cv2.setNumThreads(2)
cv2.setUseOptimized(True)

# ── LOAD MODEL ─────────────────────────────────
interp = Interpreter(MODEL_PATH, num_threads=NUM_THREADS)
interp.allocate_tensors()

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

INPUT_H = inp['shape'][2]
INPUT_W = inp['shape'][3]

IN_SCALE, IN_ZERO   = inp['quantization']
OUT_SCALE, OUT_ZERO = out['quantization']

IN_IDX  = inp['index']
OUT_IDX = out['index']

# buffers (reuse memory)
infer_buf = np.empty((1, 3, INPUT_H, INPUT_W), dtype=np.int8)
float_buf = np.empty((INPUT_H, INPUT_W, 3), dtype=np.float32)

print("Input:", inp['shape'], inp['dtype'])
print("Output:", out['shape'])

# ── SHARED STATE ───────────────────────────────
latest_jpeg = None
latest_fps  = 0.0
latest_dets = 0
lock = threading.Lock()

# ── FAST SIGMOID LUT ───────────────────────────
_SIG_X   = np.linspace(-10, 10, 1024)
_SIG_LUT = 1.0 / (1.0 + np.exp(-_SIG_X))

def fast_sigmoid(x):
    return np.interp(x, _SIG_X, _SIG_LUT)

# ── MAIN LOOP ──────────────────────────────────
def loop():
    global latest_jpeg, latest_fps, latest_dets

    cap = cv2.VideoCapture(VIDEO_PATH)

    # force lower resolution (important)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Camera failed")

    times = []

    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]

        # ── PREPROCESS (no allocations) ─────────
        cv2.resize(frame, (INPUT_W, INPUT_H),
                   dst=float_buf,
                   interpolation=cv2.INTER_NEAREST)

        float_buf /= 255.0

        img = np.transpose(float_buf, (2, 0, 1))
        img = (img / IN_SCALE + IN_ZERO).astype(np.int8)

        infer_buf[0] = img

        # ── INFERENCE ──────────────────────────
        interp.set_tensor(IN_IDX, infer_buf)
        interp.invoke()
        output = interp.get_tensor(OUT_IDX)

        output = (output.astype(np.float32) - OUT_ZERO) * OUT_SCALE

        preds = output[0].T  # (2100, 8)

        boxes = preds[:, :4]
        scores_raw = preds[:, 4:]

        # ── FAST TOP-K FILTER (no full sigmoid) ─
        top_idx = np.argsort(-scores_raw.max(axis=1))[:TOP_K]

        dets = 0

        for i in top_idx:
            raw = scores_raw[i]

            # quick reject
            if raw.max() < 0:
                continue

            probs = fast_sigmoid(raw)

            conf = probs.max()
            if conf < CONF_THRESH:
                continue

            x, y, bw, bh = boxes[i]

            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            dets += 1

        # ── FPS ────────────────────────────────
        times.append(time.time() - t0)
        if len(times) > 20:
            times.pop(0)

        fps = 1.0 / (sum(times)/len(times))

        cv2.putText(frame, f"FPS:{fps:.1f} D:{dets}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

        # ── DOWNSCALE ─────────────────────────
        if FRAME_SCALE != 1.0:
            frame = cv2.resize(frame,
                               (int(w*FRAME_SCALE), int(h*FRAME_SCALE)),
                               interpolation=cv2.INTER_NEAREST)

        # ── SKIP ENCODE IF NO DETECTIONS ───────
        if dets == 0 and latest_jpeg is not None:
            continue

        # ── JPEG ──────────────────────────────
        _, buf = cv2.imencode(".jpg", frame,
                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        with lock:
            latest_jpeg = buf.tobytes()
            latest_fps  = fps
            latest_dets = dets


# ── FLASK ─────────────────────────────────────
app = Flask(__name__)

def stream():
    last = None
    while True:
        with lock:
            jpg = latest_jpeg

        if jpg is None or jpg is last:
            time.sleep(0.001)
            continue

        last = jpg
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               jpg + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with lock:
        return jsonify(fps=round(latest_fps,1),
                       dets=latest_dets)

@app.route('/')
def index():
    return "<img src='/video_feed'>"

# ── MAIN ──────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=loop, daemon=True)
    t.start()

    while latest_jpeg is None:
        time.sleep(0.1)

    app.run(host="0.0.0.0", port=5000, threaded=True)