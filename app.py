from flask import Flask, Response
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import time

app = Flask(__name__)

VIDEO_PATH  = "walk_test.mp4"
MODEL_PATH  = "model.tflite"
INPUT_SIZE  = 320
CONF_THRESH = 0.50
NMS_THRESH  = 0.30

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

# ───────────────────────────────────────────────
#  FRAME GENERATOR
# ───────────────────────────────────────────────
def generate_frames():
    cap      = cv2.VideoCapture(VIDEO_PATH)
    fps_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
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
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# ───────────────────────────────────────────────
#  FLASK ROUTES
# ───────────────────────────────────────────────
@app.route('/')
def index():
    return (
        "<html><body style='background:#111;text-align:center;margin:0;padding:20px;'>"
        "<h1 style='color:#00ff99;font-family:Arial;'>TerrainSense Live</h1>"
        "<img src='/video_feed' style='width:100%;max-width:860px;border:2px solid #00ff99;border-radius:8px;'>"
        "</body></html>"
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("[INFO] Open browser at http://<raspberry-pi-ip>:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)