import time
import json
import struct
import numpy as np

from vl53l5cx_ctypes import VL53L5CX
from mpu6050 import mpu6050
from ahrs.filters import Madgwick

print("Initializing sensors...")

# -------- ToF Sensor --------
sensor = VL53L5CX()
sensor.set_resolution(64)
sensor.set_ranging_frequency_hz(15)
sensor.start_ranging()

# -------- IMU --------
imu = mpu6050(0x68)

# -------- Orientation filter --------
madgwick = Madgwick(frequency=100)

# quaternion state
q = np.array([1.0, 0.0, 0.0, 0.0])

print("Sensors started")

while True:

    # ---------- IMU ----------
    accel = imu.get_accel_data()
    gyro = imu.get_gyro_data()

    acc = np.array([accel['x'], accel['y'], accel['z']])
    gyr = np.radians([gyro['x'], gyro['y'], gyro['z']])

    q = madgwick.updateIMU(q, gyr=gyr, acc=acc)
    quat = q.tolist()

    # ---------- ToF ----------
    if sensor.data_ready():

        data = sensor.get_data()

        distances = list(struct.unpack("64h", data.distance_mm))
        status = list(struct.unpack("64B", bytes(data.target_status)))

        # simplify status
        status = [5 if s in (5,6,9,13) else 0 for s in status]

        packet = {
            "distances": distances,
            "status": status,
            "quat": quat
        }

        # print JSON packet
        print(json.dumps(packet), flush=True)

    time.sleep(0.01)

