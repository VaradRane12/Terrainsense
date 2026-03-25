import cv2
import numpy as np
import threading
import time
from flask import Flask, Response

# ---- CHANGE THESE ----
MODEL_PATH = "best_int8.tflite"
VIDEO_PATH = "walk_test.mp4"       # or 0 for webcam
INPUT_SIZE = 320
CONF_THRESHOLD = 0.25
CLASS_NAMES = None            # e.g. ["person", "car"] or leave None
LOOP_VIDEO = True             # restart video when it ends

# ---- IMPORT ----
try:
    from ai_edge_litert.interpreter import Interpreter
    print("Using ai_edge_litert")
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    print("Using tensorflow.lite")

# ---- LOAD MODEL ----
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape :", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

input_index  = input_details[0]['index']
output_index = output_details[0]['index']

# ---- FLASK ----
app = Flask(__name__)

frame_lock   = threading.Lock()
output_frame = None
stats        = {"fps": 0, "detections": 0, "max_conf": 0.0, "frame": 0}

# ---- INFERENCE THREAD ----
def video_loop():
    global output_frame, stats

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open: {VIDEO_PATH}")
        return

    frame_num  = 0
    fps        = 0
    counter    = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            if LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("Video ended.")
                break

        frame_num += 1
        h, w, _ = frame.shape

        # FPS
        counter += 1
        if counter >= 15:
            fps     = counter / (time.time() - start_time)
            counter = 0
            start_time = time.time()

        # ---- PREPROCESS ----
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # ---- INFERENCE ----
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)[0].T

        boxes       = output[:, :4]
        obj         = output[:, 4]
        class_probs = output[:, 5:]

        scores    = obj[:, None] * class_probs
        class_ids = np.argmax(scores, axis=1)
        conf      = np.max(scores, axis=1)

        max_conf   = float(conf.max()) if len(conf) > 0 else 0.0
        mask       = conf > CONF_THRESHOLD
        boxes      = boxes[mask]
        conf       = conf[mask]
        class_ids  = class_ids[mask]

        # ---- BOXES ----
        boxes_xyxy = []
        for b in boxes:
            x, y, bw, bh = b
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)
            boxes_xyxy.append([x1, y1, x2, y2])

        num_detections = 0

        if len(boxes_xyxy) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes_xyxy, conf.tolist(),
                score_threshold=CONF_THRESHOLD,
                nms_threshold=0.5
            )
            if len(indices) > 0:
                num_detections = len(indices.flatten())
                for i in indices.flatten():
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    cid        = class_ids[i]
                    label_name = CLASS_NAMES[cid] if CLASS_NAMES and cid < len(CLASS_NAMES) else f"class {cid}"
                    label      = f"{label_name} {conf[i]:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---- OVERLAY ----
        overlay_lines = [
            f"FPS: {fps:.1f}",
            f"Detections: {num_detections}",
            f"Max conf: {max_conf:.3f}",
            f"Frame: {frame_num}",
        ]
        for idx, line in enumerate(overlay_lines):
            cv2.putText(frame, line, (10, 30 + idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ---- DEBUG PRINT ----
        if frame_num % 30 == 1:
            print(f"[Frame {frame_num}] FPS={fps:.1f} | Max conf={max_conf:.4f} | Detections={num_detections}")

        stats = {"fps": fps, "detections": num_detections,
                 "max_conf": max_conf, "frame": frame_num}

        with frame_lock:
            output_frame = frame.copy()

        time.sleep(0.01)

    cap.release()


# ---- STREAM ----
def generate():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.02)
                continue
            _, buffer = cv2.imencode('.jpg', output_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


# ---- ROUTES ----
@app.route('/')
def index():
    s = stats
    return f"""
    <html>
    <head>
        <title>TFLite Test Stream</title>
        <meta http-equiv="refresh" content="2">
        <style>
            body {{ background: #111; color: #0f0; font-family: monospace; text-align: center; }}
            img  {{ border: 2px solid #0f0; max-width: 100%; }}
            .stats {{ font-size: 1.2em; margin: 10px; }}
        </style>
    </head>
    <body>
        <h2>TFLite Detection Test</h2>
        <div class="stats">
            FPS: {s['fps']:.1f} &nbsp;|&nbsp;
            Detections: {s['detections']} &nbsp;|&nbsp;
            Max Conf: {s['max_conf']:.3f} &nbsp;|&nbsp;
            Frame: {s['frame']}
        </div>
        <br>
        <img src="/video">
    </body>
    </html>
    """

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---- MAIN ----
if __name__ == "__main__":
    t = threading.Thread(target=video_loop, daemon=True)
    t.start()

    print("\n===========================")
    print("Open in browser:")
    print("  http://<your-pi-ip>:5000")
    print("===========================\n")

    app.run(host="0.0.0.0", port=5000, threaded=True)
