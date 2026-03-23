import time
import struct
import numpy as np
import cv2
import threading
from flask import Flask, Response, jsonify

from vl53l5cx_ctypes import VL53L5CX
from mpu6050 import mpu6050
from ahrs.filters import Madgwick
from picamera2 import Picamera2


app = Flask(__name__)

print("Initializing sensors...")

# -------- ToF (8x8 FULL) --------
sensor = VL53L5CX()
sensor.set_resolution(64)
sensor.set_ranging_frequency_hz(15)
sensor.start_ranging()

last_depth = np.zeros((8, 8))
depth_lock = threading.Lock()

# -------- IMU --------
imu = mpu6050(0x68)
madgwick = Madgwick(frequency=100)
q = np.array([1.0, 0.0, 0.0, 0.0])

quat_lock = threading.Lock()


def quaternion_to_euler(q):
    w, x, y, z = q
    t0 = 2*(w*x + y*z)
    t1 = 1 - 2*(x*x + y*y)
    roll = np.degrees(np.arctan2(t0, t1))

    t2 = 2*(w*y - z*x)
    t2 = np.clip(t2, -1, 1)
    pitch = np.degrees(np.arcsin(t2))

    t3 = 2*(w*z + x*y)
    t4 = 1 - 2*(y*y + z*z)
    yaw = np.degrees(np.arctan2(t3, t4))

    return roll, pitch, yaw


# -------- CAMERA --------
width, height = 320, 240
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (width, height)})
)
picam2.start()


# -------- THREADS --------
def tof_thread():
    global last_depth
    while True:
        if sensor.data_ready():
            data = sensor.get_data()
            raw = np.array(struct.unpack("64h", data.distance_mm)).reshape(8, 8)

            with depth_lock:
                last_depth = raw


def imu_thread():
    global q
    while True:
        accel = imu.get_accel_data()
        gyro = imu.get_gyro_data()

        acc = np.array([accel['x'], accel['y'], accel['z']])
        gyr = np.radians([gyro['x'], gyro['y'], gyro['z']])

        q = madgwick.updateIMU(q, gyr=gyr, acc=acc)

        with quat_lock:
            pass


threading.Thread(target=tof_thread, daemon=True).start()
threading.Thread(target=imu_thread, daemon=True).start()


# -------- VIDEO STREAM --------
def generate_frames():
    while True:
        frame = picam2.capture_array()

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        with depth_lock:
            distances = last_depth.copy()

        with quat_lock:
            roll, pitch, yaw = quaternion_to_euler(q)

        # process depth
        depth_resized = cv2.resize(distances, (width, height), interpolation=cv2.INTER_NEAREST)
        depth_resized = np.nan_to_num(depth_resized, nan=2000)

        depth_clipped = np.clip(depth_resized, 200, 1500)
        depth_uint8 = ((depth_clipped - 200) / (1500 - 200) * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        # rotate
        M = cv2.getRotationMatrix2D((width//2, height//2), -roll, 1)
        heatmap = cv2.warpAffine(heatmap, M, (width, height))

        overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

        _, buffer = cv2.imencode('.jpg', overlay)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# -------- ROUTES --------
@app.route('/')
def index():
    return """
    <h2>ToF Heatmap Stream</h2>
    <img src="/video">
    <p><a href="/depth">Raw 8x8 Depth JSON</a></p>
    """


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/depth')
def depth():
    with depth_lock:
        d = last_depth.tolist()

    with quat_lock:
        quat = q.tolist()

    return jsonify({
        "depth_8x8": d,
        "quat": quat
    })


# -------- RUN --------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
