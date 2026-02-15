import time
import board
import busio
import adafruit_vl53l1x

i2c = busio.I2C(board.SCL, board.SDA)
tof = adafruit_vl53l1x.VL53L1X(i2c)

tof.distance_mode = 2
tof.timing_budget = 100
tof.start_ranging()

while True:
    print(tof.distance)
    time.sleep(0.2)
