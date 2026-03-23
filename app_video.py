import cv2
import numpy as np
import threading
from flask import Flask, Response
import tensorflow as tf

app = Flask(__name__)

# -------------------- LOAD MODEL --------------------
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SIZE = input_details[0]['shape'][1]

# -------------------- GLOBAL FRAME --------------------
frame_lock = threading.Lock()
output_frame = None

# -------------------- NMS --------------------
def nms(boxes, scores, iou_threshold=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        if len(idxs) == 1:
            break

        rest = idxs[1:]

        xx1 = np.maximum(boxes[i][0], boxes[rest][:, 0])
        yy1 = np.maximum(boxes[i][1], boxes[rest][:, 1])
        xx2 = np.minimum(boxes[i][2], boxes[rest][:, 2])
        yy2 = np.minimum(boxes[i][3], boxes[rest][:, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        area1 = (boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1])
        area2 = (boxes[rest][:,2]-boxes[rest][:,0]) * (boxes[rest][:,3]-boxes[rest][:,1])

        iou = inter / (area1 + area2 - inter + 1e-6)

        idxs = idxs[1:][iou < iou_threshold]

    return keep

# -------------------- INFERENCE THREAD --------------------
def camera_loop():
    global output_frame

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape

        # preprocess
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # decode
        output = output.T  # (2100, 8)

        boxes = output[:, :4]
        obj = output[:, 4]
        class_probs = output[:, 5:]

        scores = obj[:, None] * class_probs
        class_ids = np.argmax(scores, axis=1)
        conf = np.max(scores, axis=1)

        mask = conf > 0.4

        boxes = boxes[mask]
        class_ids = class_ids[mask]
        conf = conf[mask]

        # convert boxes (xywh → xyxy)
        boxes_xyxy = []
        for b in boxes:
            x, y, bw, bh = b
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)
            boxes_xyxy.append([x1, y1, x2, y2])

        if len(boxes_xyxy) > 0:
            boxes_xyxy = np.array(boxes_xyxy)
            keep = nms(boxes_xyxy, conf)

            for i in keep:
                x1, y1, x2, y2 = boxes_xyxy[i]
                label = f"{class_ids[i]} {conf[i]:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # update global frame
        with frame_lock:
            output_frame = frame.copy()

# -------------------- STREAM --------------------
def generate():
    global output_frame

    while True:
        with frame_lock:
            if output_frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return "<img src='/video'>"

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------- MAIN --------------------
if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    print("Flask running on http://<pi-ip>:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)