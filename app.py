import time
import threading
from flask import Flask, Response, jsonify
from pipeline import AsyncPipeline

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH  = "best_int8.tflite"   # path to your tflite model
CAMERA_SRC  = 0                    # 0 = first camera, or "/dev/video0"
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
JPEG_QUALITY= 70                   # 60–80 is the sweet spot for Pi streaming

app      = Flask(__name__)
pipeline = AsyncPipeline(MODEL_PATH, source=CAMERA_SRC,
                         conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH)

# ── Routes ────────────────────────────────────────────────────────────────

def generate_frames():
    while True:
        jpeg = pipeline.get_jpeg(quality=JPEG_QUALITY)
        if jpeg is None:
            time.sleep(0.02)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify({
        "fps": round(pipeline.current_fps, 1),
        "detections": pipeline.detection_count,
    })

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Terrainsense</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #0a0a0a;
      color: #e0e0e0;
      font-family: "Courier New", monospace;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
    }
    h1 {
      font-size: 1.2rem;
      letter-spacing: 0.3em;
      color: #00ff80;
      margin-bottom: 16px;
      text-transform: uppercase;
    }
    #stream-wrap {
      border: 1px solid #00ff8044;
      border-radius: 4px;
      overflow: hidden;
      max-width: 100%;
      box-shadow: 0 0 30px #00ff8022;
    }
    #stream-wrap img { display: block; max-width: 100%; }
    #stats {
      margin-top: 14px;
      display: flex;
      gap: 24px;
      font-size: 0.85rem;
      color: #888;
    }
    #stats span { color: #00ff80; font-weight: bold; }
    #fps-val, #det-val { color: #00ff80; }
  </style>
</head>
<body>
  <h1>Terrainsense // Live</h1>
  <div id="stream-wrap">
    <img src="/video_feed" alt="stream">
  </div>
  <div id="stats">
    <div>FPS &nbsp;<span id="fps-val">--</span></div>
    <div>DETECTIONS &nbsp;<span id="det-val">--</span></div>
  </div>
  <script>
    setInterval(async () => {
      try {
        const r = await fetch('/stats');
        const d = await r.json();
        document.getElementById('fps-val').textContent = d.fps;
        document.getElementById('det-val').textContent = d.detections;
      } catch(e) {}
    }, 1000);
  </script>
</body>
</html>'''

# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting pipeline...")
    pipeline.start()
    print("Pipeline running. Open http://<pi-ip>:5000")
    # threaded=False — inference is already async, no need for Flask threading
    app.run(host='0.0.0.0', port=5000, threaded=False)
