from flask import Flask, Response
import time, math, cv2
import numpy as np

# ---------------- CAMERA (Picamera2) ----------------
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)
picam2.start()
time.sleep(0.2)

# ---------------- MPU6050 ----------------
from smbus2 import SMBus

MPU_ADDR = 0x68
ACCEL_XOUT_H = 0x3B

bus = SMBus(1)

# Hard reset MPU
bus.write_byte_data(MPU_ADDR, 0x6B, 0x80)
time.sleep(0.1)
bus.write_byte_data(MPU_ADDR, 0x6B, 0x00)
time.sleep(0.1)

def read_word(reg):
    try:
        h = bus.read_byte_data(MPU_ADDR, reg)
        l = bus.read_byte_data(MPU_ADDR, reg + 1)
        v = (h << 8) | l
        return v - 65536 if v > 32767 else v
    except OSError:
        return None

def read_pitch():
    ax_raw = read_word(ACCEL_XOUT_H)
    az_raw = read_word(ACCEL_XOUT_H + 4)

    if ax_raw is None or az_raw is None:
        return None

    ax = ax_raw / 16384.0
    az = az_raw / 16384.0
    return math.degrees(math.atan2(ax, az))

# ---------------- TOF ----------------
import board, busio
import adafruit_vl53l1x

i2c = busio.I2C(board.SCL, board.SDA)
tof = adafruit_vl53l1x.VL53L1X(i2c)
tof.distance_mode = 2
tof.timing_budget = 100
tof.start_ranging()
time.sleep(0.1)

# ---------------- FLASK ----------------
app = Flask(__name__)

# ROI settings
W, H = 640, 480
roi_w, roi_h = 180, 140
roi_x = W // 2 - roi_w // 2
roi_y = H // 2 - roi_h // 2

# MPU cache
last_pitch = 0.0
last_pitch_time = 0.0

def generate_frames():
    global last_pitch, last_pitch_time

    while True:
        # ---- Rate-limited MPU read (20 Hz) ----
        now = time.time()
        if now - last_pitch_time > 0.05:
            p = read_pitch()
            if p is not None:
                last_pitch = p
            last_pitch_time = now

        pitch = last_pitch

        # ---- Camera frame ----
        frame = picam2.capture_array()

        # ---- TOF ----
        dist = tof.distance

        # ---- Decision logic ----
        label = "SAFE"
        color = (0, 255, 0)

        if dist < 600 and abs(pitch) < 10:
            label = "OBSTACLE"
            color = (0, 0, 255)
        elif dist > 1500:
            label = "DROP"
            color = (255, 0, 0)
        elif abs(pitch) > 8:
            label = "SLOPE"
            color = (0, 165, 255)

        # ---- Overlay ----
        cv2.rectangle(
            frame,
            (roi_x, roi_y),
            (roi_x + roi_w, roi_y + roi_h),
            color, 2
        )

        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPL EX, 1.2, color, 3)

        cv2.putText(frame, f"Dist: {dist:.1f} mm", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255,  255), 2)

        cv2.putText(frame, f"Pitch: {pitch:.1f} deg", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        _, jpg = cv2.imencode(".jpg", frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg.tobytes() +
            b"\r\n"
        )

@app.route("/")
def index():
    return """
    <html>
      <head><title>Obstacle Mapping Demo</title></head>
      <body style="background:#111;color:white;text-align:center;">
        <h2>Camera + ToF + IMU Obstacle Map</h2>
        <img src="/video" width="640" height="480">
      </body>
    </html>
    """

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=False)
