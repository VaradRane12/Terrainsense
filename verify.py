import cv2
import numpy as np
import onnxruntime as ort
import time

# ───────────────────────────────────────────────
#  CONFIG
# ───────────────────────────────────────────────
VIDEO_PATH  = "walk_test.mp4"
MODEL_PATH  = "best.onnx"
INPUT_SIZE  = 320
CONF_THRESH = 0.50    # ⬆️ raised to reduce false detections
NMS_THRESH  = 0.30    # ⬇️ aggressive NMS to kill overlaps

CLASS_LABELS = {
    0: ("OBSTACLE", (0, 165, 255)),
    1: ("PERSON",   (255, 0, 0)),
    2: ("POTHOLE",  (0, 0, 255)),
    3: ("VEHICLE",  (0, 255, 255)),
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ───────────────────────────────────────────────
#  LOAD MODEL
# ───────────────────────────────────────────────
print("[INFO] Loading model...")
session    = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("[INFO] ✅ Model loaded!")

# ───────────────────────────────────────────────
#  PREPROCESS
# ───────────────────────────────────────────────
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

# ───────────────────────────────────────────────
#  POSTPROCESS — ONE best detection only
# ───────────────────────────────────────────────
def postprocess(output, frame):
    h_frame, w_frame = frame.shape[:2]

    preds        = output[0].squeeze(0).T
    boxes_xywh   = preds[:, :4]
    class_scores = sigmoid(preds[:, 4:])      # apply sigmoid
    confidences  = np.max(class_scores, axis=1)
    class_ids    = np.argmax(class_scores, axis=1)

    # Filter low confidence
    mask = confidences > CONF_THRESH
    if not np.any(mask):
        return None, None, 0.0

    boxes_f   = boxes_xywh[mask]
    confs_f   = confidences[mask].tolist()
    classes_f = class_ids[mask]

    # Convert xywh → xyxy
    scale_x = w_frame / INPUT_SIZE
    scale_y = h_frame / INPUT_SIZE

    boxes_xyxy = []
    for box in boxes_f:
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * scale_x)
        y1 = int((cy - bh / 2) * scale_y)
        x2 = int((cx + bw / 2) * scale_x)
        y2 = int((cy + bh / 2) * scale_y)
        x1 = max(0, x1);       y1 = max(0, y1)
        x2 = min(w_frame, x2); y2 = min(h_frame, y2)
        boxes_xyxy.append([x1, y1, x2, y2])

    # Aggressive NMS
    boxes_nms = [[x1, y1, x2-x1, y2-y1] for x1,y1,x2,y2 in boxes_xyxy]
    indices   = cv2.dnn.NMSBoxes(boxes_nms, confs_f, CONF_THRESH, NMS_THRESH)

    if len(indices) == 0:
        return None, None, 0.0

    # ── Pick ONLY the single highest confidence box ──
    best_i    = max(indices.flatten(), key=lambda i: confs_f[i])
    x1, y1, x2, y2 = boxes_xyxy[best_i]
    conf      = float(confs_f[best_i])
    label, color = CLASS_LABELS.get(int(classes_f[best_i]), ("UNKNOWN", (255,255,255)))

    # Draw single clean bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Label tag
    text        = f"{label}  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1-th-12), (x1+tw+8, y1), color, -1)
    cv2.putText(frame, text, (x1+4, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return label, color, conf

# ───────────────────────────────────────────────
#  MAIN LOOP
# ───────────────────────────────────────────────
cap      = cv2.VideoCapture(VIDEO_PATH)
fps_list = []

print("[INFO] Starting... Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    t1     = time.time()
    output = session.run(None, {input_name: preprocess(frame)})
    t2     = time.time()

    fps = 1.0 / (t2 - t1 + 1e-6)
    fps_list.append(fps)
    avg_fps = sum(fps_list[-10:]) / len(fps_list[-10:])

    result = postprocess(output, frame)

    if result[0] is None:
        label, color, conf = "ALL CLEAR", (0, 255, 0), 0.0
    else:
        label, color, conf = result

    # ── Clean HUD ────────────────────────────────
    # Alert banner at top
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)   # black bar
    cv2.putText(frame, f"⚠ {label}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

    # Confidence + FPS bottom left
    cv2.putText(frame, f"Conf: {conf:.2f}  |  FPS: {avg_fps:.1f}",
                (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("TerrainSense", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Avg FPS: {sum(fps_list)/len(fps_list):.1f}")