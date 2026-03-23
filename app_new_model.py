import cv2
import numpy as np
import threading
import time
from flask import Flask, Response
import tensorflow as tf

# ───────── CONFIG ─────────
MODEL_PATH = "best_int8.tflite"
INPUT_SIZE = 320
CONF_THRESH = 0.5
TOP_K = 15
FRAME_SKIP = 2
JPEG_QUALITY = 60

# ───────── LOAD MODEL ─────────
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_scale, in_zero = input_details[0]['quantization']
out_scale, out_zero = output_details[0]['quantization']

# ───────── GLOBAL STATE ─────────
frame_lock = threading.Lock()
latest_frame = None
latest_jpeg = None

# ───────── PREPROCESS ─────────
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE),
                     interpolation=cv2.INTER_NEAREST)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # HWC → CHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # INT8 quantize
    img = (img / in_scale + in_zero).astype(np.int8)

    return img

# ───────── POSTPROCESS ─────────
def postprocess(output, frame):
    h, w, _ = frame.shape

    # dequantize
    output = (output.astype(np.float32) - out_zero) * out_scale

    preds = output[0].T   # (2100, 8)

    boxes = preds[:, :4]
    scores = preds[:, 4:]

    # sigmoid
    scores = 1 / (1 + np.exp(-scores))

    conf = scores.max(axis=1)
    cls  = scores.argmax(axis=1)

    # filter
    idx = np.argsort(-conf)[:TOP_K]

    for i in idx:
        if conf[i] < CONF_THRESH:
            continue

        x, y, bw, bh = boxes[i]

        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    return frame

# ───────── CAMERA THREAD ─────────
def capture_loop():
    global latest_frame

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        with frame_lock:
            latest_frame = frame

# ───────── INFERENCE THREAD ─────────
def inference_loop():
    global latest_frame, latest_jpeg

    count = 0
    times = []

    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        count += 1
        if count % FRAME_SKIP != 0:
            continue

        t0 = time.time()

        inp = preprocess(frame)

        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()

        out = interpreter.get_tensor(output_details[0]['index'])

        frame = postprocess(out, frame)

        # FPS calc
        times.append(time.time() - t0)
        if len(times) > 20:
            times.pop(0)

        fps = 1.0 / (sum(times)/len(times))

        cv2.putText(frame, f"FPS:{fps:.1f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

        # encode
        _, buf = cv2.imencode(".jpg", frame,
                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        latest_jpeg = buf.tobytes()

# ───────── FLASK ─────────
app = Flask(__name__)

def stream():
    last = None
    while True:
        if latest_jpeg is None or latest_jpeg == last:
            time.sleep(0.005)
            continue

        last = latest_jpeg

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               latest_jpeg + b'\r\n')

@app.route('/')
def index():
    return "<img src='/video'>"

@app.route('/video')
def video():
    return Response(stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

# ───────── MAIN ─────────
if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()

    while latest_jpeg is None:
        time.sleep(0.1)

    app.run(host="0.0.0.0", port=5000, threaded=True)