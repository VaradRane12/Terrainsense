import time
import struct
import numpy as np
import cv2

from vl53l5cx_ctypes import VL53L5CX
from mpu6050 import mpu6050
from ahrs.filters import Madgwick
from picamera2 import Picamera2


print("Initializing sensors...")

# -------- ToF --------
sensor = VL53L5CX()
sensor.set_resolution(64)
sensor.set_ranging_frequency_hz(15)
sensor.start_ranging()

# -------- IMU --------
imu = mpu6050(0x68)
madgwick = Madgwick(frequency=100)
q = np.array([1.0, 0.0, 0.0, 0.0])

# -------- Camera --------
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (640, 480)})
)
picam2.start()

print("Sensors started")


def quaternion_to_euler(q):
    w, x, y, z = q

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x*x + y*y)
    roll = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y*y + z*z)
    yaw = np.degrees(np.arctan2(t3, t4))

    return roll, pitch, yaw


while True:

    # -------- CAMERA --------
    frame = picam2.capture_array()

    # Fix RGBA → BGR if needed
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # -------- IMU --------
    accel = imu.get_accel_data()
    gyro = imu.get_gyro_data()

    acc = np.array([accel['x'], accel['y'], accel['z']])
    gyr = np.radians([gyro['x'], gyro['y'], gyro['z']])

    q = madgwick.updateIMU(q, gyr=gyr, acc=acc)
    roll, pitch, yaw = quaternion_to_euler(q)

    # -------- TOF --------
    if sensor.data_ready():

        data = sensor.get_data()

        distances = np.array(
            struct.unpack("64h", data.distance_mm)
        ).reshape(8, 8)

        status = list(struct.unpack("64B", bytes(data.target_status)))

        # mask invalid readings
        mask = np.array([1 if s in (5,6,9,13) else 0 for s in status]).reshape(8, 8)
        distances = np.where(mask == 1, distances, np.nan)

        # -------- RESIZE --------
        depth_resized = cv2.resize(
            distances,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # -------- NORMALIZE --------
        depth_norm = cv2.normalize(
            depth_resized, None, 0, 255, cv2.NORM_MINMAX
        )
        depth_uint8 = np.nan_to_num(depth_norm).astype(np.uint8)

        # -------- HEATMAP --------
        heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        # Ensure same size
        if heatmap.shape[:2] != frame.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        # Ensure 3-channel
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if len(heatmap.shape) == 2:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)

        # -------- IMU ROTATION --------
        h, w = heatmap.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), -roll, 1)
        heatmap = cv2.warpAffine(heatmap, M, (w, h))

        # -------- OVERLAY --------
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        # -------- DISPLAY --------
        cv2.imshow("ToF Heatmap Overlay", overlay)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(0.01)

cv2.destroyAllWindows()
