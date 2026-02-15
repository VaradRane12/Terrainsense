import time
import math
import numpy as np

# --------- SMBUS FIRST (MPU6050) ---------
from smbus2 import SMBus

MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B

bus = SMBus(1)

# Wake MPU6050 BEFORE Blinka touches I2C
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)
time.sleep(0.1)

# --------- NOW IMPORT BLINKA / TOF ---------
import board
import busio
import adafruit_vl53l1x

i2c = busio.I2C(board.SCL, board.SDA)

tof = adafruit_vl53l1x.VL53L1X(i2c)
tof.distance_mode = 1
tof.timing_budget = 50

print("Sensors initialised correctly")

# --------- MPU READ FUNCTIONS ---------
def read_word(reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)
    val = (high << 8) | low
    if val > 32767:
        val -= 65536
    return val

def read_pitch():
    ax = read_word(ACCEL_XOUT_H) / 16384.0
    az = read_word(ACCEL_XOUT_H + 4) / 16384.0
    return math.degrees(math.atan2(ax, az))

# --------- CALIBRATION ---------
print("Calibrating… keep still")

dist_samples = []
pitch_samples = []

for _ in range(20):
    dist_samples.append(tof.distance)
    pitch_samples.append(read_pitch())
    time.sleep(0.05)

base_dist = np.mean(dist_samples)
base_pitch = np.mean(pitch_samples)

print(f"Baseline distance: {base_dist:.1f} mm")
print(f"Baseline pitch: {base_pitch:.1f} deg")

# --------- MAIN LOOP ---------
try:
    t = 0
    while True:
        dist = tof.distance
        pitch = read_pitch()

        delta_dist = dist - base_dist
        delta_pitch = pitch - base_pitch

        label = "FLAT / SAFE"
        if delta_dist > 120 and abs(delta_pitch) < 5:
            label = "STEP DOWN / DROP"
        elif delta_dist < -120:
            label = "STEP UP / KERB"
        elif abs(delta_pitch) > 7:
            label = "RAMP / SLOPE"

        print(
            f"t={t:03d} | Dist={dist:6.1f} mm | "
            f"ΔDist={delta_dist:6.1f} | "
            f"Pitch={pitch:6.1f}° | {label}"
        )


        t += 1
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped cleanly")
