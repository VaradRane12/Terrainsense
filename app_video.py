"""
Optimized single-file YOLOv8 TFLite Flask streaming app.
Inference runs in background thread — Flask only serves latest JPEG.
Run: python app_single.py
Then open http://<pi-ip>:5000
"""

import cv2
import numpy as np
import time
import threading
from tflite_runtime.interpreter import Interpreter
from flask import Flask, Response, jsonify

# ── CONFIG ────────────────────────────────────────────────────────────────
MODEL_PATH   = "best_int8.tflite"
VIDEO_PATH   = "walk_test.mp4"    # change to 0 for webcam later
CONF_THRESH  = 0.25
IOU_THRESH   = 0.45
NUM_THREADS  = 4                   # uses all 4 Pi cores for XNNPACK
JPEG_QUALITY = 65                  # lower = faster encode, less bandwidth
FRAME_SCALE  = 0.5                 # shrink display frame before encode
# ─────────────────────────────────────────────────────────────────────────

# ── Load model ────────────────────────────────────────────────────────────
print("Loading model...")
interp = Interpreter(MODEL_PATH, num_threads=NUM_THREADS)
interp.allocate_tensors()

inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
INPUT_H = inp_det['shape'][1]
INPUT_W = inp_det['shape'][2]
INP_IDX = inp_det['index']
OUT_IDX = out_det['index']

# Pre-allocate inference buffer once — never reallocated
infer_buf = np.empty((1, INPUT_H, INPUT_W, 3), dtype=np.float32)

print(f"Input : {inp_det['shape']}  dtype={inp_det['dtype']}")
print(f"Output: {out_det['shape']}")
print(f"Model ready. Input: {INPUT_W}x{INPUT_H}")

# ── Shared state between inference thread and Flask ───────────────────────
latest_jpeg = None
latest_fps  = 0.0
latest_dets = 0
frame_lock  = threading.Lock()

# ── Sigmoid LUT — computed once at startup ────────────────────────────────
_SIG_X   = np.linspace(-10, 10, 2048)
_SIG_LUT = 1.0 / (1.0 + np.exp(-_SIG_X))

def fast_sigmoid(x):
    return np.interp(x, _SIG_X, _SIG_LUT)


# ── Background inference thread ───────────────────────────────────────────
def inference_loop():
    global latest_jpeg, latest_fps, latest_dets

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {VIDEO_PATH}")
    print(f"Video opened: {VIDEO_PATH}")

    fps_times  = []
    frame_num  = 0

    while True:
        t_frame = time.time()

        ret, frame = cap.read()
        if not ret:
            # Loop video back to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        orig_h, orig_w = frame.shape[:2]

        # ── Preprocess into pre-allocated buffer (no new allocation) ──────
        resized = cv2.resize(frame, (INPUT_W, INPUT_H),
                             interpolation=cv2.INTER_LINEAR)
        np.copyto(infer_buf[0], resized.astype(np.float32) * (1.0 / 255.0))

        # ── Inference ─────────────────────────────────────────────────────
        t_inf = time.time()
        interp.set_tensor(INP_IDX, infer_buf)
        interp.invoke()
        output = interp.get_tensor(OUT_IDX)
        inf_ms = (time.time() - t_inf) * 1000

        # ── Postprocess ───────────────────────────────────────────────────
        preds      = output[0]           # [8, 2100]
        boxes_raw  = preds[:4, :].T      # [2100, 4]
        scores_raw = preds[4:, :].T      # [2100, num_classes]

        # Logit-space pre-filter: sigmoid only on high-confidence candidates
        logit_thresh = np.log(CONF_THRESH / (1.0 - CONF_THRESH + 1e-9))
        mask     = scores_raw.max(axis=1) > logit_thresh
        num_dets = 0

        if mask.any():
            scores = fast_sigmoid(scores_raw[mask])
            fb     = boxes_raw[mask]
            fc     = scores.max(axis=1)
            fi     = scores.argmax(axis=1)

            cx, cy, w, h = fb[:,0], fb[:,1], fb[:,2], fb[:,3]
            x1 = ((cx - w/2) * orig_w / INPUT_W).clip(0, orig_w).astype(int)
            y1 = ((cy - h/2) * orig_h / INPUT_H).clip(0, orig_h).astype(int)
            x2 = ((cx + w/2) * orig_w / INPUT_W).clip(0, orig_w).astype(int)
            y2 = ((cy + h/2) * orig_h / INPUT_H).clip(0, orig_h).astype(int)
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            indices = cv2.dnn.NMSBoxes(
                boxes_xyxy.tolist(), fc.tolist(), CONF_THRESH, IOU_THRESH
            )

            if len(indices) > 0:
                num_dets = len(indices)
                for idx in indices.flatten():
                    bx1, by1, bx2, by2 = boxes_xyxy[idx]
                    score = fc[idx]
                    label = f"cls{fi[idx]}: {score:.2f}"
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 80), 2)
                    (lw, lh), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame,
                                  (bx1, by1 - lh - 8), (bx1 + lw + 4, by1),
                                  (0, 255, 80), -1)
                    cv2.putText(frame, label, (bx1 + 2, by1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # ── FPS tracking ──────────────────────────────────────────────────
        fps_times.append(time.time() - t_frame)
        if len(fps_times) > 30:
            fps_times.pop(0)
        fps = 1.0 / (sum(fps_times) / len(fps_times))

        # ── HUD overlay ───────────────────────────────────────────────────
        info = f"FPS:{fps:.1f}  inf:{inf_ms:.0f}ms  dets:{num_dets}"
        cv2.putText(frame, info, (8, orig_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, info, (8, orig_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 1)

        # ── Downscale before encode to cut JPEG encode time ───────────────
        if FRAME_SCALE != 1.0:
            disp = cv2.resize(frame,
                              (int(orig_w * FRAME_SCALE),
                               int(orig_h * FRAME_SCALE)),
                              interpolation=cv2.INTER_LINEAR)
        else:
            disp = frame

        # ── Encode to JPEG ────────────────────────────────────────────────
        _, buf      = cv2.imencode('.jpg', disp,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        jpeg_bytes  = buf.tobytes()

        # ── Push to shared state ──────────────────────────────────────────
        with frame_lock:
            latest_jpeg = jpeg_bytes
            latest_fps  = fps
            latest_dets = num_dets

        frame_num += 1
        if frame_num % 30 == 0:
            print(f"FPS: {fps:.1f}  inf: {inf_ms:.0f}ms  dets: {num_dets}    ",
                  end="\r")


# ── Flask app ─────────────────────────────────────────────────────────────
app = Flask(__name__)

def generate():
    """
    Serves latest_jpeg without ever blocking on inference.
    Tracks last served pointer — only yields when a NEW frame is ready.
    """
    last_served = None
    while True:
        with frame_lock:
            jpeg = latest_jpeg

        if jpeg is None or jpeg is last_served:
            time.sleep(0.005)   # tiny sleep prevents CPU spin
            continue

        last_served = jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    with frame_lock:
        return jsonify(fps=round(latest_fps, 1), dets=latest_dets)


@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Terrainsense</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      background:#0d0d0d;
      display:flex; flex-direction:column;
      align-items:center; justify-content:center;
      min-height:100vh; gap:14px;
      font-family:"Courier New", monospace;
    }
    h2 { color:#00ff80; letter-spacing:.25em; font-size:1rem; margin-top:16px; }
    #wrap {
      border:1px solid #00ff8044;
      box-shadow:0 0 24px #00ff8018;
      line-height:0;
    }
    #wrap img { max-width:95vw; display:block; }
    #stats { color:#555; font-size:.8rem; margin-bottom:16px; }
    #stats span { color:#00ff80; }
  </style>
</head>
<body>
  <h2>TERRAINSENSE // LIVE</h2>
  <div id="wrap"><img src="/video_feed"></div>
  <div id="stats">
    FPS <span id="fps">--</span> &nbsp;|&nbsp; DETECTIONS <span id="dets">--</span>
  </div>
  <script>
    setInterval(async () => {
      try {
        const r = await fetch('/stats');
        const d = await r.json();
        document.getElementById('fps').textContent  = d.fps;
        document.getElementById('dets').textContent = d.dets;
      } catch(e) {}
    }, 800);
  </script>
</body>
</html>'''


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    print("Waiting for first inference...")
    while latest_jpeg is None:
        time.sleep(0.1)
    print("First frame ready.\n")

    print("Open http://<pi-ip>:5000\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)