import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter as tflite
from flask import Flask, Response

# ─── CONFIG ───────────────────────────────────────────
MODEL_PATH     = "best_float16.tflite"
CLASS_NAMES    = ["obstacle", "person", "pothole", "vehical"]
CLASS_COLORS   = {
    0: (0, 165, 255),    # obstacle → orange
    1: (0, 255, 0),      # person   → green
    2: (0, 0, 255),      # pothole  → red
    3: (255, 0, 0),      # vehical  → blue
}
CONF_THRESHOLD = 0.5
INPUT_SIZE     = 640
JPEG_QUALITY   = 35
SKIP_FRAMES    = 3
CAM_WIDTH      = 640
CAM_HEIGHT     = 480
# ──────────────────────────────────────────────────────

app = Flask(__name__)

# Load model
interpreter = tflite(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img

def postprocess(output, orig_frame):
    predictions = output[0].T  # (8400, 8)
    h, w = orig_frame.shape[:2]

    for pred in predictions:
        cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence < CONF_THRESHOLD:
            continue

        x1 = int((cx - bw / 2) * w / INPUT_SIZE)
        y1 = int((cy - bh / 2) * h / INPUT_SIZE)
        x2 = int((cx + bw / 2) * w / INPUT_SIZE)
        y2 = int((cy + bh / 2) * h / INPUT_SIZE)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        label = f"{CLASS_NAMES[class_id]}: {confidence:.2f}"
        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return orig_frame

def generate_frames():
    # CSI camera - try index 0 first
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ CSI camera not found, trying index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("❌ No camera found!")
        return

    print("✅ CSI Camera connected!")
    # rest of loop stays same...

@app.route('/')
def index():
    return '''
        <html><body style="background:#000;margin:0">
        <img src="/video" style="width:100%;height:100vh;object-fit:contain">
        </body></html>
    '''

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("✅ Open browser at http://<pi-ip>:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)