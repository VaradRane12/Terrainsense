"""
sensor_stream.py
────────────────
Subprocess that reads from ToF (VL53L5CX 8×8) and IMU (MPU6050),
applies Madgwick orientation filter, and outputs JSON lines.
"""

import time
import json
import struct
import sys
import numpy as np

from vl53l5cx_ctypes import VL53L5CX
from mpu6050 import mpu6050
from ahrs.filters import Madgwick

print("[SENSOR_STREAM] Initializing sensors...", file=sys.stderr, flush=True)

# -------- ToF Sensor --------
try:
    sensor = VL53L5CX()
    sensor.set_resolution(64)
    sensor.set_ranging_frequency_hz(15)
    sensor.start_ranging()
    print("[SENSOR_STREAM] ToF sensor initialized", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[SENSOR_STREAM] ToF initialization failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

# -------- IMU --------
try:
    imu = mpu6050(0x68)
    print("[SENSOR_STREAM] IMU initialized", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[SENSOR_STREAM] IMU initialization failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

# -------- Orientation filter --------
madgwick = Madgwick(frequency=100)

# quaternion state
q = np.array([1.0, 0.0, 0.0, 0.0])

print("[SENSOR_STREAM] All sensors started", file=sys.stderr, flush=True)

try:
    while True:

        # ---------- IMU ----------
        try:
            accel = imu.get_accel_data()
            gyro = imu.get_gyro_data()

            acc = np.array([accel['x'], accel['y'], accel['z']])
            gyr = np.radians([gyro['x'], gyro['y'], gyro['z']])

            q = madgwick.updateIMU(q, gyr=gyr, acc=acc)
            quat = q.tolist()
        except Exception as e:
            print(f"[SENSOR_STREAM] IMU read error: {e}", file=sys.stderr, flush=True)
            quat = [1.0, 0.0, 0.0, 0.0]

        # ---------- ToF ----------
        if sensor.data_ready():
            try:
                data = sensor.get_data()

                distances = list(struct.unpack("64h", data.distance_mm))
                status = list(struct.unpack("64B", bytes(data.target_status)))

                # simplify status: only keep values 5, 6, 9, 13 as valid (5), others as invalid (0)
                status = [5 if s in (5, 6, 9, 13) else 0 for s in status]

                packet = {
                    "distances": distances,
                    "status": status,
                    "quat": quat
                }

                # output JSON packet to stdout (parent process reads this)
                print(json.dumps(packet), flush=True)
            except Exception as e:
                print(f"[SENSOR_STREAM] ToF read error: {e}", file=sys.stderr, flush=True)

        # Keep latency low; ToF data_ready() gates real update rate.
        time.sleep(0.005)

except KeyboardInterrupt:
    print("[SENSOR_STREAM] Shutting down...", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[SENSOR_STREAM] Fatal error: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
