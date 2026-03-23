import cv2
import numpy as np
import threading
import time
from flask import Flask, Response
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

# -------------------- LOAD MODEL --------------------
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']

INPUT_SIZE = 256  # reduced from 320 for speed

# -------------------- GLOBAL FRAME --------------------
frame_lock = threading.Lock()
output_frame = None

# -------------------- INFERENCE THREAD --------------------
def camera_loop():
    global output_frame

    cap = cv2.VideoCapture("walk_test.mp4")  # change if needed

    if not cap.isOpened():
        print("Video failed to open")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        # -------- SKIP EVERY 2ND FRAME --------
        if frame_count % 2 != 0:
            with frame_lock:
                output_frame = frame.copy()
            continue

        h, w, _ = frame.shape

        # -------- PREPROCESS --------
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # -------- INFERENCE --------
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)[0]

        output = output.T  # (N, 8)

        boxes = output[:, :4]
        obj = output[:, 4]
        class_probs = output[:, 5:]

        scores = obj[:, None] * class_probs
        class_ids = np.argmax(scores, axis=1)
        conf = np.max(scores, axis=1)

        mask = conf > 0.4
        boxes = boxes[mask]
        conf = conf[mask]
        class_ids = class_ids[mask]

        boxes_xyxy = []

        for b in boxes:
            x, y, bw, bh = b
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)
            boxes_xyxy.append([x1, y1, x2, y2])

        # -------- FAST NMS --------
        if len(boxes_xyxy) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes_xyxy,
                conf.tolist(),
                score_threshold=0.4,
                nms_threshold=0.5
            )

            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    label = f"{class_ids[i]} {conf[i]:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # -------- OUTPUT --------
        with frame_lock:
            output_frame = frame.copy()

# -------------------- STREAM --------------------
def generate():
    global output_frame

    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.01)
                continue

            _, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.02)

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return "<h2>Video Stream</h2><img src='/video'>"

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------- MAIN --------------------
if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    print("Running on http://<pi-ip>:5000/video")
    app.run(host="0.0.0.0", port=5000, threaded=True)