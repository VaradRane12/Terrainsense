import time
import struct
import numpy as np
import cv2
import threading

from vl53l5cx_ctypes import VL53L5CX
from mpu6050 import mpu6050
from ahrs.filters import Madgwick
from picamera2 import Picamera2


print("Initializing sensors...")

# -------- ToF (FAST CONFIG) --------
sensor = VL53L5CX()
sensor.set_resolution(16)              # request 4x4 mode
sensor.set_ranging_frequency_hz(30)    # faster updates
sensor.start_ranging()

# shared depth buffer
last_depth = np.zeros((4, 4))
depth_lock = threading.Lock()


def tof_thread():
    global last_depth
    while True:
        if sensor.data_ready():
            data = sensor.get_data()

            # ALWAYS read 64 values
            raw = np.array(struct.unpack("64h", data.distance_mm)).reshape(8, 8)

            # downsample 8x8 → 4x4
            distances = raw.reshape(4, 2, 4, 2).mean(axis=(1, 3))

            with depth_lock:
                last_depth = distances


# start ToF thread
threading.Thread(target=tof_thread, daemon=True).start()


# -------- IMU --------
imu = mpu6050(0x68)
madgwick = Madgwick(frequency=100)
q = np.array([1.0, 0.0, 0.0, 0.0])


def quaternion_to_euler(q):
    w, x, y, z = q

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x*x + y*y)
    roll = np.degrees(np.arctan2(t0, t1))

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(t2))

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.degrees(np.arctan2(t3, t4))

    return roll, pitch, yaw


# -------- CAMERA --------
width, height = 320, 240
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (width, height)})
)
picam2.start()


# -------- VIDEO OUTPUT --------
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_fast.avi", fourcc, 20, (width, height))

print("Running fast pipeline → output_fast.avi")


while True:

    # -------- CAMERA --------
    frame = picam2.capture_array()

    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # -------- IMU --------
    accel = imu.get_accel_data()
    gyro = imu.get_gyro_data()

    acc = np.array([accel['x'], accel['y'], accel['z']])
    gyr = np.radians([gyro['x'], gyro['y'], gyro['z']])

    q = madgwick.updateIMU(q, gyr=gyr, acc=acc)
    roll, pitch, yaw = quaternion_to_euler(q)

    # -------- GET LATEST DEPTH --------
    with depth_lock:
        distances = last_depth.copy()

    # -------- PROCESS --------
    depth_resized = cv2.resize(
        distances,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    depth_resized = np.nan_to_num(depth_resized, nan=2000)

    # fixed range scaling (important)
    depth_clipped = np.clip(depth_resized, 200, 1500)
    depth_uint8 = ((depth_clipped - 200) / (1500 - 200) * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    # -------- IMU STABILIZATION --------
    M = cv2.getRotationMatrix2D((width//2, height//2), -roll, 1)
    heatmap = cv2.warpAffine(heatmap, M, (width, height))

    # -------- OVERLAY --------
    overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    # -------- SAVE --------
    out.write(overlay)
