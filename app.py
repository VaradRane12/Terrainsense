from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)
VIDEO_PATH = "footpath.mp4"

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        roi = frame[int(h*0.6):h, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)

        edge_density = np.count_nonzero(edges) / (edges.size)

        if edge_density < 0.01:
            label = "POSSIBLE DROP / STEP"
            color = (0, 0, 255)
        elif edge_density < 0.03:
            label = "OBSTACLE / UNEVEN"
            color = (0, 165, 255)
        else:
            label = "FLAT / SAFE"
            color = (0, 255, 0)

        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
