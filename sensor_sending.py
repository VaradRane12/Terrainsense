import time
import json
import struct
from vl53l5cx_ctypes import VL53L5CX

print("Initializing VL53L5CX...")

sensor = VL53L5CX()

sensor.set_resolution(64)        # 8×8 mode
sensor.set_ranging_frequency_hz(15)

sensor.start_ranging()

print("Sensor started")
while True:
    if sensor.data_ready():
        data = sensor.get_data()

        distances = list(struct.unpack("64h", data.distance_mm))
        status = list(struct.unpack("64B", bytes(data.target_status)))  # fixed

        quat = [1, 0, 0, 0]

        packet = {
            "distances": distances,
            "status": status,
            "quat": quat
        }

        print(json.dumps(packet), flush=True)

    time.sleep(0.05)